# Proxima — guia do projeto (para o Claude e para mim)

> Projeto 3 do portfólio. Sucessor do **Bedrock** (storage engine, projeto 1, concluído).
> Nome: **Proxima** (do latim *proximus*, "o mais próximo" — a busca por vizinho mais próximo;
> e *Proxima Centauri*, a estrela mais próxima). O roadmap detalhado em degraus está em `ROADMAP.md`.

## Ideia principal
Proxima é um **vector database / engine de busca por similaridade**, escrito em **Rust**, feito
como projeto de portfólio-bandeira de **infraestrutura de IA**. O objetivo é o mesmo do Bedrock:
**aprender sistemas implementando** — agora no domínio que mais valoriza backend hoje. Armazena
**embeddings** (vetores de alta dimensão) e responde **k-nearest-neighbors** ("os K vetores mais
parecidos com este") em escala, com **busca aproximada (ANN)** via índice **HNSW**.

O que é: um banco que indexa vetores e responde queries de similaridade rápido, com persistência
em disco, API HTTP, e um **trade-off recall×latência medido e documentado**. É a espinha dorsal de
RAG, busca semântica, recomendação e memória de agentes — e por baixo **é um banco de dados**, então
reaproveita tudo que aprendi no Bedrock (storage em disco, durabilidade, concorrência).

Por que impressiona: mostra que entendo a fundação dos sistemas de IA, não que "treino modelos".
Constrói a infra que a IA precisa em produção — exatamente o trabalho de um AI/backend engineer.

Conceitos que o projeto prova: distância em alta dimensão (cosseno, L2), brute-force kNN como
baseline honesta, **grafo HNSW** (multicamadas, small-world, navegação greedy), o trade-off
**recall@k × latência × memória**, quantização de vetores, persistência do índice, e benchmark
contra baseline (FAISS/hnswlib).

Diferencial visual planejado (degrau tardio): **visualizador 3D do grafo** (projeção do espaço
de alta dimensão para 3D) — chama atenção de recrutador e prova que entendo a estrutura.

## Workspace Cargo (proposto)
- `crates/core` — tipos base: `Vector`, métricas de distância, `VectorStore` (persistência)
- `crates/index` — o índice ANN: brute-force primeiro, depois HNSW
- `crates/server` — API HTTP (mesma postura "à mão sobre std::net" do Bedrock)
- `crates/bench` — harness de recall×latência + carregador de datasets (SIFT/GloVe)
- `playground/viz` — futuro visualizador 3D (degrau tardio; isolado do core)

> Só `crates/core` está criado (ponto de partida do degrau 1). Os demais entram no degrau em que
> são necessários — adicione ao `members` do `Cargo.toml` quando chegar a hora.

## Decisões de design já firmadas
- **Brute-force primeiro, HNSW depois.** O kNN exato (força bruta) é a baseline de **recall=100%**
  contra a qual medir o HNSW. Sem ele, não há como provar a qualidade do índice aproximado. É o
  análogo do "SQLite como baseline" do Bedrock.
- **Reaproveitar o Bedrock como camada de persistência** dos vetores (o storage durável já existe).
  O índice HNSW é a camada nova por cima. Isso fecha o arco "construí o storage, depois o vector DB".
  (Pré-requisito no Bedrock: uma API de valores em **bytes** — hoje os valores são `String`. Ver o
  `IMPROVEMENTS.md` do Bedrock.)
- **Métrica configurável** (cosseno e L2/euclidiana) — mesma interface, implementações diferentes.
- **Sem framework de ML.** Eu não gero embeddings nem treino nada; o engine **recebe** vetores
  prontos. Para testes uso datasets públicos com ground-truth (SIFT1M, GloVe).
- **Sem `unsafe` gratuito.** SIMD/otimização agressiva só depois de medir, e com justificativa.

---

## Como o Claude deve trabalhar comigo (IMPORTANTE — herdado do Bedrock)
- Eu sou **aluno**; o Claude é **instrutor**. Nível de Rust: **intermediário** (subindo).
- **NÃO escrever o código de implementação por mim.** O papel do Claude é: explicar conceitos,
  especificar funções (assinatura + o que faz + decisões/edge cases), apontar armadilhas, e **me
  deixar implementar**. Pode escrever scaffolding/testes não-centrais e revisar meu código.
- Trechos de ≤3 linhas para ilustrar sintaxe de Rust são ok. Implementação completa de uma função
  central (ex.: o `search` do HNSW, a inserção no grafo), **não** — é justamente o que eu preciso suar.
- Quando eu travar, me dê pistas e perguntas que me empurrem, não a resposta pronta. ~30-45 min de
  luta antes de socorro. HNSW é difícil: espere que eu erre a navegação greedy e o select-neighbors
  algumas vezes — me deixe depurar com pistas.
- Sempre ser **honesto sobre as garantias e os números reais**. Toda afirmação de performance DEVE
  vir com a condição (qual dataset, qual dimensão, quais parâmetros M/ef, recall medido em quê).
  Nada de "é rápido" solto: rápido a que recall, contra que baseline, em que hardware.

---

## Onde paramos (atualizar a cada sessão)
- ⏳ PROJETO NÃO INICIADO. Repo só com o scaffold (workspace + `crates/core` vazio + docs + notas).
  Próximo passo: **degrau 1** (tipos base `Vector` + trait `Metric` com L2/cosseno + testes).
- (a cada sessão, registrar aqui o degrau fechado, a garantia/número real atual, e o próximo passo)

## Garantias / números a documentar (honestidade obrigatória)
Como no Bedrock: declarar explicitamente o que o engine faz e **sob quais condições**. Para um
vector DB isso significa:
- **Recall@k medido**, não prometido: "recall@10 = 0.98 no SIFT1M, dim=128, M=16, ef_search=128".
- **Latência por query** (p50/p95/p99) e **throughput** (QPS), com o recall correspondente.
- **Custo de memória** do índice (HNSW é faminto de RAM — ser honesto sobre isso).
- **Exatidão da baseline**: o brute-force é recall=1.0 por definição; o HNSW é aproximado.
- O que NÃO faz (no estado atual): filtros por metadata, deleção real, distribuição, etc.

---

## Definição de "pronto" do projeto 3 (o nível que impressiona recrutador)
Os mesmos quatro entregáveis que fecharam o Bedrock, adaptados:

1. **Benchmarks publicados** — curva **recall×latência** (varrendo `ef_search`), QPS, memória, e
   comparação honesta contra **FAISS ou hnswlib** no mesmo dataset. Plotar, não só tabelar.
2. **Correção demonstrada** — teste reproduzível que mede recall@k do HNSW contra o ground-truth do
   dataset (ou contra o brute-force) e **afirma um piso** (ex.: "recall@10 ≥ 0.95"). É o análogo do
   crash-recovery do Bedrock: a prova objetiva de que o índice faz o que diz.
3. **README em inglês** explicando **decisões e trade-offs**: por que HNSW e não IVF/LSH/árvore-kd,
   por que o grafo em camadas funciona, o trade-off recall×latência×memória, e os limites do engine.
4. **CI rodando** — build + testes + clippy + fmt a cada push, e idealmente o benchmark de recall
   rodando como teste de regressão (falha se o recall cair abaixo do piso).

> [!important]
> O Claude deve manter esses quatro no radar e **não me deixar declarar "pronto"** sem eles — em
> especial o #2 (recall medido) e o #1 (curva contra FAISS). Um vector DB sem número de recall é só
> uma alegação. Profundidade na execução é o que contrata.

## Horizonte futuro (NÃO trabalhar agora — só contexto)
Depois do core impecável: **(a)** o **visualizador 3D do grafo** (projeção UMAP/PCA para 3D, mostrar
as camadas e a navegação da query — forte apelo visual no portfólio); **(b)** integração no
**projeto 4** (pipeline RAG usa este engine); **(c)** quantização de produto (PQ) para reduzir
memória; **(d)** filtros por metadata (filterable HNSW). Tudo isso é degrau **posterior** ao "pronto".

> [!warning]
> Mesma regra do Bedrock: **profundidade sobre novidade**. Não pular pro visualizador 3D ou pra PQ
> antes do core (brute-force + HNSW + recall medido + FAISS baseline) estar no nível "pronto". O 3D
> é vitrine; o recall medido é a substância. Se me vir querendo fazer o 3D antes da curva de recall,
> me lembrar disto.

---

## Livro de referência — DDIA + os papers
- **DDIA** toca pouco em ANN diretamente, mas o vocabulário de **índices** (cap. 3 "Storage and
  Retrieval") e de **trade-offs de leitura/escrita/memória** vale. Citar capítulo/seção **pelo nome**,
  nunca número de página (mesma regra do Bedrock).
- **Paper canônico do HNSW**: Malkov & Yashunin, *"Efficient and robust approximate nearest neighbor
  search using Hierarchical Navigable Small World graphs"* (2016/2018). É a fonte da verdade do
  algoritmo. Quando eu implementar a navegação e o select-neighbors, ancore no paper.
- **ANN-Benchmarks** (Bernhardsson) — a referência de como se mede recall×QPS de forma honesta.

---

## Anotações de estudo (formato obrigatório — idêntico ao Bedrock)
- Toda explicação/conceito vira um `.md` em `notas/` (gitignored, Obsidian). Um arquivo por sessão,
  em ordem (`01-...`), kebab-case. As notas são meu guia de revisão: devo reconstruir o raciocínio
  **sem reabrir o código**.
- Template fixo de cada nota: (1) cabeçalho com links; (2) objetivo da sessão; (3) conceitos, cada um
  em subseção, no meu nível, com analogia e citação do DDIA/paper pelo nome; (4) especificação das
  funções (assinatura + o que faz + edge cases, **sem** implementação completa); (5) pegadinhas de
  Rust; (6) bugs que cometi; (7) estado ao fim + número/garantia real atual + próximo passo.
- Callouts do Obsidian (`> [!note]`, `> [!warning]`, `> [!tip]`, `> [!important]`), tabelas para
  comparações (HNSW vs IVF vs brute-force; cosseno vs L2), blocos com linguagem marcada.
- Toda afirmação de performance DEVE dizer a **condição** (dataset, dim, M, ef, recall medido).
- Ao fim da sessão: atualizar `00-indice.md` e o bloco "Onde paramos" deste arquivo.
