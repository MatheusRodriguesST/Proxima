# Proxima — roadmap em degraus (vector database / ANN engine em Rust)

> Mesma filosofia do Bedrock: cada degrau resolve a **limitação concreta** do anterior, e você
> implementa (o Claude instrui, não escreve o core). A ordem é deliberada — você só sente por que
> o HNSW é necessário depois de medir a dor do brute-force. "Começar simples, medir, refinar."

## Princípio que organiza tudo

Um vector DB tem duas metades: **(1) guardar vetores** com durabilidade (você já sabe fazer — é o
Bedrock) e **(2) buscar os mais parecidos rápido** (o desafio novo — o índice ANN). O roadmap
constrói a metade 1 reaproveitando o Bedrock, prova a busca com força bruta (baseline de recall
perfeito), e só então ataca o HNSW — medindo o ganho contra essa baseline a cada passo.

---

## FASE A — Fundamentos: vetores, distância e busca exata

### Degrau 1 — Tipos base + métricas de distância
**Limitação que resolve:** ponto de partida; sem isso não há o que comparar.

O que construir: o tipo `Vector` (um array de `f32` de dimensão fixa), e as funções de distância
**L2 (euclidiana)** e **cosseno**. Decisão de design: a métrica é uma escolha do índice, então
modele como um trait (ex.: `trait Metric { fn distance(a, b) -> f32; }`) com duas implementações.

Conceitos: por que distância em alta dimensão (o "significado" de um embedding vira proximidade
geométrica); a diferença entre L2 e cosseno (cosseno ignora magnitude, mede ângulo — padrão para
embeddings de texto normalizados); a maldição da dimensionalidade (por que isso fica difícil).

Edge cases a tratar: dimensões incompatíveis (erro, não panic em produção), vetor zero no cosseno
(divisão por zero), normalização.

Entregável: testes unitários das métricas com valores conhecidos à mão.

### Degrau 2 — Brute-force kNN (a baseline de recall = 100%)
**Limitação que resolve:** sem busca, não há banco. E sem busca *exata*, não há como medir o quão
boa é a busca *aproximada* depois.

O que construir: `fn knn(query, k) -> Vec<(id, distance)>` que compara a query com **todos** os
vetores, ordena, e retorna os K menores. É O(N) e lento — de propósito. Este é o oráculo: por
definição tem recall perfeito, então é contra ele (ou contra o ground-truth do dataset) que você
mede o HNSW.

Conceitos: por que isso não escala (O(N·d) por query — inviável em milhões de vetores); por que
ainda assim é essencial (ground-truth, e o fallback correto para datasets pequenos — vários DBs
reais fazem full-scan abaixo de um limiar).

> [!tip]
> Use um heap de tamanho K (`BinaryHeap`) em vez de ordenar tudo — é o primeiro contato com a
> otimização "não faça trabalho desnecessário" que o HNSW leva ao extremo.

### Degrau 3 — Persistência (reaproveitar o Bedrock)
**Limitação que resolve:** até aqui os vetores vivem só na RAM; somem ao reiniciar.

O que construir: usar o **Bedrock** como camada de storage durável dos vetores (chave = id, valor =
bytes do vetor). Carregar no boot, persistir nas inserções. Aqui você colhe o investimento do
projeto 1: durabilidade a crash já está resolvida.

Conceitos: serialização de vetores para bytes (formato binário, como você já fez no Bedrock);
separar "o store durável dos vetores" do "índice de busca" (o índice pode ser reconstruído a partir
do store — importante para o HNSW, que é caro de persistir).

Entregável: inserir N vetores, reiniciar o processo, confirmar que a busca ainda funciona.

---

## FASE B — O coração: índice HNSW

> [!important]
> Esta é a fase difícil e o cerne do projeto. Ancore no **paper do Malkov & Yashunin (2016)**.
> Espere suar — a navegação greedy e o select-neighbors são onde quase todo mundo erra. Implemente
> incrementalmente e teste o recall a cada sub-degrau contra o brute-force do degrau 2.

### Degrau 4 — Grafo navegável de camada única (NSW, sem hierarquia)
**Limitação que resolve:** o brute-force é O(N). Um grafo de proximidade permite navegar até os
vizinhos sem visitar todos.

O que construir: um grafo onde cada nó (vetor) tem arestas para alguns vizinhos próximos, e uma
**busca greedy**: comece em um nó de entrada, vá sempre para o vizinho mais próximo da query, pare
no mínimo local. Inserção: conecte o novo nó aos M mais próximos encontrados.

Conceitos: a propriedade "small-world" (poucos saltos conectam quaisquer dois nós); a busca greedy e
seu calcanhar — fica presa em **mínimos locais** (motiva a hierarquia do próximo degrau); a estrutura
de dados central da busca: uma lista dinâmica de candidatos de tamanho `ef` (o "feixe" de exploração).

Parâmetro que nasce aqui: **`ef_search`** — tamanho da fila de candidatos durante a busca. Maior =
explora mais = mais recall, mais lento. É o botão de recall×latência em tempo de query.

### Degrau 5 — Hierarquia: o H do HNSW
**Limitação que resolve:** o grafo de camada única cai em mínimos locais e navega devagar em escala.
As camadas resolvem isso dando "atalhos de longo alcance".

O que construir: múltiplas camadas. Cada vetor entra numa camada sorteada (probabilidade decai
logaritmicamente — a maioria só na camada 0; poucos sobem). A busca começa no topo (poucos nós,
saltos longos — as "rodovias"), desce camada a camada refinando até a camada 0 (todos os nós — o
"endereço final").

Conceitos: a analogia das rodovias→ruas→endereço; por que a probabilidade logarítmica dá O(log N)
empírico na busca; o nó de entrada fixo no topo. Conectar com **skip lists** (a mesma ideia de
níveis probabilísticos, citada no paper).

Parâmetros que nascem aqui:
- **`M`** — máximo de arestas por nó por camada (na camada 0, geralmente 2M). Maior = grafo mais
  denso = mais recall, mais memória, build mais lento. Típico: 16.
- **`ef_construction`** — como o `ef_search`, mas durante a inserção. Maior = grafo melhor construído
  = mais recall, build mais lento. Típico: 100–200. **Não afeta o tamanho do índice**, só a qualidade.

> [!warning]
> O **select-neighbors heurístico** (escolher quais M vizinhos manter, não só os M mais próximos, para
> evitar arestas redundantes e manter o grafo conectado) é a parte mais sutil do paper. Faça primeiro
> a versão ingênua (M mais próximos), meça o recall, e só então implemente a heurística e meça de novo
> o ganho. É a lição "começar grosso, medir, refinar" do Bedrock aplicada ao algoritmo.

### Degrau 6 — Tuning e a curva recall × latência
**Limitação que resolve:** você tem o HNSW, mas não *provou* que ele é bom. Sem número, é só alegação.

O que construir: um harness de benchmark que, num dataset com ground-truth (**SIFT1M** ou **GloVe**),
varre `ef_search` e plota **recall@k × QPS**. Mostrar a curva: como mais exploração compra mais recall
ao custo de latência. Medir também o custo de memória do índice e o tempo de build vs `M`/`ef_construction`.

Conceitos: **recall@k** (fração dos K verdadeiros vizinhos que o índice achou); por que o trade-off é
fundamental e não um bug; como ler uma curva recall×QPS (o canto superior direito é o ideal).

> [!important]
> Este degrau é o **#2 da definição de pronto** (correção demonstrada). É a prova objetiva. Sem a
> curva de recall, o projeto não está pronto, por mais bonito que o código seja.

---

## FASE C — Banco de verdade: rede, concorrência, escala

### Degrau 7 — API HTTP (mesma postura do Bedrock)
**Limitação que resolve:** até aqui é uma lib; um banco precisa aceitar clientes.

O que construir: servidor HTTP/1.1 à mão sobre `std::net` (mesma postura "sem framework" do Bedrock):
`POST /vectors` (inserir id + vetor), `POST /search` (query + k + ef_search → resultados),
`GET /stats` (introspecção). Thread-por-conexão sobre o índice compartilhado.

Conceitos: serialização dos vetores no corpo (JSON é aceitável aqui na fronteira, mesmo que o storage
seja binário — separa o protocolo do formato em disco); validação de dimensão na borda.

### Degrau 8 — Concorrência
**Limitação que resolve:** múltiplos clientes ao mesmo tempo. Você já sabe o padrão do Bedrock.

O que construir: leituras (search) concorrentes XOR escrita (insert), começando com `RwLock` grosso
como no Bedrock 8a. Conceito novo: inserção no HNSW **muta o grafo** (adiciona arestas em nós
existentes), então a seção crítica de escrita é mais delicada que num KV store — pensar no que pode
ser lido enquanto uma inserção reescreve a vizinhança de um nó.

> [!tip]
> Reuse o aprendizado do Bedrock (positioned I/O, `Arc<RwLock>`, "começar grosso e refinar"). Aqui o
> refinamento lock-free é ainda mais valioso porque search é o caminho quente.

### Degrau 9 — Quantização (memória) — opcional, alto valor
**Limitação que resolve:** HNSW é faminto de RAM; vetores `f32` em dimensão alta dominam a memória.

O que construir: **quantização escalar** (f32→int8) como primeiro passo, medindo o impacto no recall.
Mencionar (e talvez implementar) **product quantization (PQ)** como o passo avançado. Sempre medindo:
quantização **reduz recall** — mostrar o trade-off memória×recall com números.

Conceitos: por que PQ é o que torna bilhões de vetores viáveis; o trade-off compressão×precisão.

---

## FASE D — Vitrine (só depois do core "pronto")

### Degrau 10 — Visualizador 3D do grafo
**Limitação que resolve:** nenhuma técnica — é apelo visual e prova de entendimento. Mas só vale
depois que recall/latência/FAISS-baseline estiverem feitos.

O que construir: projetar os vetores de alta dimensão para 3D (PCA ou UMAP), renderizar o grafo HNSW
em camadas (Three.js no `playground/viz`, isolado do core em Rust), e **animar uma query** descendo
as camadas. Mostra visualmente as "rodovias→ruas→endereço".

> [!warning]
> Isto é vitrine, não substância. Não comece antes do #1–#4 da definição de pronto. Um 3D lindo sobre
> um índice sem recall medido não engana um revisor técnico.

---

## Resumo dos degraus

| # | Degrau | Resolve a limitação de... | Fase |
|---|--------|---------------------------|------|
| 1 | Tipos + métricas (L2, cosseno) | nada (ponto de partida) | A |
| 2 | Brute-force kNN | não ter baseline de recall | A |
| 3 | Persistência (via Bedrock) | vetores só na RAM | A |
| 4 | Grafo navegável 1 camada (NSW) | brute-force O(N) | B |
| 5 | Hierarquia (HNSW completo) | mínimos locais / escala | B |
| 6 | Curva recall×latência + FAISS | não ter provado a qualidade | B |
| 7 | API HTTP | ser só uma lib | C |
| 8 | Concorrência | um cliente por vez | C |
| 9 | Quantização (opcional) | RAM faminta do HNSW | C |
| 10 | Visualizador 3D | (vitrine, não técnica) | D |

## Ordem de prioridade para "pronto que impressiona"
Degraus **1→6 são o núcleo inegociável** — é o que prova que você entende ANN de verdade (o degrau 6,
a curva de recall contra FAISS, é o coração da credibilidade). Com 1–6 + README + CI você já tem um
projeto-bandeira de IA-infra. Os degraus 7–8 (banco de rede concorrente) elevam para "sistema", e
9–10 são diferenciais. Não troque a profundidade dos 1–6 pela vitrine do 10.

*Roadmap criado em junho/2026. Parâmetros típicos (M≈16, ef_construction≈100–200, ef_search≈50–200)
e ferramentas de referência (FAISS, hnswlib, ANN-Benchmarks, datasets SIFT/GloVe) evoluem — reconfira
o paper do HNSW e o estado da arte ao executar a Fase B.*
