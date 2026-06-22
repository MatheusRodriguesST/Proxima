//! Interactive 2D visualizer for the Proxima engine.
//!
//! Drives the real `proxima-index` (insert / update / remove / search). A node's
//! on-screen position **is** its vector, so "nearest on screen" matches the
//! engine's nearest exactly; moving a node updates its value. The motion is a
//! smooth glide toward each node's coordinate — purely cosmetic, never changing
//! the data.
//!
//! Controls:
//!
//! ```text
//! P                   run a k-NN search from the cursor
//! A                   add a node at the cursor
//! left drag node      move a node (updates its vector)
//! left drag empty     pan
//! wheel               zoom to cursor
//! Delete / Backspace  remove the selected node
//! + / -               neighbors per node, k
//! M                   toggle metric (L2 / cosine)
//! Esc                 clear search / selection
//! R                   reset the view
//! ```
//!
//! Run: `cargo run -p proxima-viz`

use macroquad::prelude::*;
use proxima_core::{Cosine, Vector, L2};
use proxima_index::{BruteForceIndex, Neighbor};
use std::collections::{BTreeSet, HashMap};

const LERP_SPEED: f32 = 9.0;
const NODE_RADIUS: f32 = 18.0;

fn window_conf() -> Conf {
    Conf {
        window_title: "Proxima — interactive vector graph".to_owned(),
        window_width: 1280,
        window_height: 820,
        high_dpi: true,
        ..Default::default()
    }
}

struct VNode {
    id: u64,
    label: String,
    pos: Vec2,
    target: Vec2, // the node's vector; `pos` glides toward it

    radius: f32,
    component: usize,
}

struct Search {
    pos: Vec2,
    results: Vec<Neighbor>,
    scanned: usize,
    age: f32,
}

struct App {
    index: BruteForceIndex,
    nodes: Vec<VNode>,
    edges: Vec<(usize, usize)>,
    components: usize,
    k: usize,
    use_cosine: bool,
    dirty: bool,
    cam_offset: Vec2,
    zoom: f32,
    dragging: Option<usize>,
    panning: bool,
    selected: Option<usize>,
    last_mouse: Vec2,
    search: Option<Search>,
}

impl App {
    fn new() -> Self {
        let mut app = App {
            index: BruteForceIndex::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            components: 0,
            k: 3,
            use_cosine: false,
            dirty: true,
            cam_offset: vec2(screen_width() * 0.5, screen_height() * 0.5),
            zoom: 1.0,
            dragging: None,
            panning: false,
            selected: None,
            last_mouse: Vec2::ZERO,
            search: None,
        };
        // Words grouped by meaning, as an embedding model would place them.
        let vocab = [
            ("cat", -300.0, -150.0),
            ("dog", -225.0, -160.0),
            ("lion", -310.0, -85.0),
            ("tiger", -230.0, -90.0),
            ("wolf", -270.0, -35.0),
            ("apple", 300.0, -150.0),
            ("banana", 225.0, -160.0),
            ("grape", 310.0, -85.0),
            ("orange", 230.0, -90.0),
            ("lemon", 270.0, -35.0),
            ("rust", -300.0, 150.0),
            ("vector", -225.0, 160.0),
            ("index", -310.0, 215.0),
            ("search", -230.0, 220.0),
            ("graph", -270.0, 265.0),
            ("star", 300.0, 150.0),
            ("orbit", 225.0, 160.0),
            ("galaxy", 310.0, 215.0),
            ("comet", 230.0, 220.0),
            ("nebula", 270.0, 265.0),
        ];
        for (label, x, y) in vocab {
            app.add_node(vec2(x, y), label.to_string());
        }
        // Bloom intro: start everyone at the center and let them glide outward.
        for node in &mut app.nodes {
            node.pos = Vec2::ZERO;
        }
        app
    }

    fn to_screen(&self, world: Vec2) -> Vec2 {
        world * self.zoom + self.cam_offset
    }

    fn to_world(&self, screen: Vec2) -> Vec2 {
        (screen - self.cam_offset) / self.zoom
    }

    fn add_node(&mut self, world: Vec2, label: String) {
        let id = self.index.insert(Vector::from([world.x, world.y]));
        self.nodes.push(VNode {
            id,
            label,
            pos: world,
            target: world,
            radius: 0.0,
            component: 0,
        });
        self.dirty = true;
    }

    fn remove_selected(&mut self) {
        if let Some(i) = self.selected.take() {
            self.index.remove(self.nodes[i].id);
            self.nodes.swap_remove(i);
            self.dragging = None;
            self.search = None;
            self.dirty = true;
        }
    }

    fn search_vec(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        if self.use_cosine {
            self.index.search(&Cosine, query, k)
        } else {
            self.index.search(&L2, query, k)
        }
    }

    fn run_search(&mut self, world: Vec2) {
        let results = self.search_vec(&[world.x, world.y], self.k);
        self.search = Some(Search {
            pos: world,
            results,
            scanned: self.index.len(),
            age: 0.0,
        });
    }

    /// Recompute k-NN edges (via the engine) and connected components.
    fn rebuild(&mut self) {
        let id_to_idx: HashMap<u64, usize> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id, i))
            .collect();

        let mut set: BTreeSet<(usize, usize)> = BTreeSet::new();
        for n in &self.nodes {
            let vector = match self.index.get(n.id) {
                Some(v) => v,
                None => continue,
            };
            let i = id_to_idx[&n.id];
            for nb in self.search_vec(vector, self.k + 1) {
                if nb.id == n.id {
                    continue;
                }
                if let Some(&j) = id_to_idx.get(&nb.id) {
                    set.insert((i.min(j), i.max(j)));
                }
            }
        }
        self.edges = set.into_iter().collect();
        self.assign_components();
        self.dirty = false;
    }

    fn assign_components(&mut self) {
        let n = self.nodes.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }
        for &(a, b) in &self.edges {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra] = rb;
            }
        }
        let mut root_color: HashMap<usize, usize> = HashMap::new();
        for i in 0..n {
            let r = find(&mut parent, i);
            let next = root_color.len();
            let c = *root_color.entry(r).or_insert(next);
            self.nodes[i].component = c;
        }
        self.components = root_color.len();
    }

    fn node_at_screen(&self, screen: Vec2) -> Option<usize> {
        self.nodes
            .iter()
            .position(|n| self.to_screen(n.pos).distance(screen) <= (n.radius * self.zoom).max(8.0))
    }

    fn handle_input(&mut self) {
        let m = mouse_position();
        let mouse = vec2(m.0, m.1);

        let (_, wheel) = mouse_wheel();
        if wheel != 0.0 {
            let factor = if wheel > 0.0 { 1.1 } else { 1.0 / 1.1 };
            self.cam_offset = mouse - (mouse - self.cam_offset) * factor;
            self.zoom = (self.zoom * factor).clamp(0.2, 4.0);
        }

        if is_mouse_button_pressed(MouseButton::Left) {
            match self.node_at_screen(mouse) {
                Some(i) => {
                    self.dragging = Some(i);
                    self.selected = Some(i);
                }
                None => {
                    self.panning = true;
                    self.selected = None;
                }
            }
        }
        if is_mouse_button_down(MouseButton::Left) {
            if let Some(i) = self.dragging {
                // A node's position is its value: moving it updates the vector.
                let world = self.to_world(mouse);
                self.nodes[i].pos = world;
                self.nodes[i].target = world;
                self.index
                    .update(self.nodes[i].id, Vector::from([world.x, world.y]));
                self.dirty = true;
            } else if self.panning {
                self.cam_offset += mouse - self.last_mouse;
            }
        }
        if is_mouse_button_released(MouseButton::Left) {
            self.dragging = None;
            self.panning = false;
        }

        if is_key_pressed(KeyCode::P) {
            self.run_search(self.to_world(mouse));
        }
        if is_key_pressed(KeyCode::A) {
            let label = format!("v{}", self.nodes.len());
            self.add_node(self.to_world(mouse), label);
        }
        if is_key_pressed(KeyCode::Delete) || is_key_pressed(KeyCode::Backspace) {
            self.remove_selected();
        }
        if is_key_pressed(KeyCode::Equal) || is_key_pressed(KeyCode::KpAdd) {
            self.k = (self.k + 1).min(self.nodes.len().saturating_sub(1).max(1));
            self.dirty = true;
        }
        if is_key_pressed(KeyCode::Minus) || is_key_pressed(KeyCode::KpSubtract) {
            self.k = self.k.saturating_sub(1).max(1);
            self.dirty = true;
        }
        if is_key_pressed(KeyCode::M) {
            self.use_cosine = !self.use_cosine;
            self.dirty = true;
            self.search = None;
        }
        if is_key_pressed(KeyCode::Escape) {
            self.search = None;
            self.selected = None;
        }
        if is_key_pressed(KeyCode::R) {
            self.cam_offset = vec2(screen_width() * 0.5, screen_height() * 0.5);
            self.zoom = 1.0;
        }

        self.last_mouse = mouse;
    }

    fn update(&mut self) {
        let dt = get_frame_time().min(1.0 / 30.0);
        self.handle_input();
        if self.dirty {
            self.rebuild();
        }
        let t = (dt * LERP_SPEED).min(1.0);
        for node in &mut self.nodes {
            node.pos += (node.target - node.pos) * t;
            node.radius += (NODE_RADIUS - node.radius) * t;
        }
        if let Some(s) = &mut self.search {
            s.age += dt;
        }
    }

    fn draw(&self) {
        clear_background(Color::new(0.04, 0.05, 0.07, 1.0));
        self.draw_edges();
        self.draw_search_overlay();
        self.draw_nodes();
        self.draw_search_marker();
        self.draw_hud();
        self.draw_results_panel();
    }

    fn draw_edges(&self) {
        for &(a, b) in &self.edges {
            let ca = component_color(self.nodes[a].component);
            let p1 = self.to_screen(self.nodes[a].pos);
            let p2 = self.to_screen(self.nodes[b].pos);
            draw_line(p1.x, p1.y, p2.x, p2.y, 2.0, with_alpha(ca, 0.45));
        }
    }

    fn draw_nodes(&self) {
        for (i, node) in self.nodes.iter().enumerate() {
            let color = component_color(node.component);
            let p = self.to_screen(node.pos);
            let r = node.radius * self.zoom;
            draw_circle(p.x, p.y, r * 2.1, with_alpha(color, 0.05));
            draw_circle(p.x, p.y, r * 1.45, with_alpha(color, 0.10));
            draw_circle(p.x, p.y, r, color);
            let rim = if self.selected == Some(i) {
                Color::new(1.0, 1.0, 1.0, 0.95)
            } else {
                Color::new(1.0, 1.0, 1.0, 0.35)
            };
            draw_circle_lines(p.x, p.y, r, 2.0, rim);
            let fs = 15.0;
            let d = measure_text(&node.label, None, fs as u16, 1.0);
            draw_text(
                &node.label,
                p.x - d.width * 0.5,
                p.y + d.height * 0.5,
                fs,
                WHITE,
            );
        }
    }

    fn draw_search_overlay(&self) {
        let Some(s) = &self.search else { return };
        let q = self.to_screen(s.pos);
        let result_ids: BTreeSet<u64> = s.results.iter().map(|n| n.id).collect();
        // faint line to every node — the brute-force scan compared them all
        for node in &self.nodes {
            if result_ids.contains(&node.id) {
                continue;
            }
            let p = self.to_screen(node.pos);
            draw_line(q.x, q.y, p.x, p.y, 1.0, Color::new(0.5, 0.55, 0.65, 0.08));
        }
        let pulse = 0.5 + 0.5 * (s.age * 5.0).sin();
        for (rank, nb) in s.results.iter().enumerate() {
            if let Some(node) = self.nodes.iter().find(|n| n.id == nb.id) {
                let p = self.to_screen(node.pos);
                draw_line(q.x, q.y, p.x, p.y, 2.5, Color::new(1.0, 0.85, 0.3, 0.85));
                let ring = node.radius * self.zoom + 5.0 + pulse * 5.0;
                draw_circle_lines(p.x, p.y, ring, 2.5, Color::new(1.0, 0.85, 0.3, 0.9));
                let tag = format!("#{}  {:.2}", rank + 1, nb.distance);
                draw_text(
                    &tag,
                    p.x + ring,
                    p.y - ring,
                    15.0,
                    Color::new(1.0, 0.9, 0.5, 1.0),
                );
            }
        }
    }

    fn draw_search_marker(&self) {
        let Some(s) = &self.search else { return };
        let q = self.to_screen(s.pos);
        let pulse = 0.5 + 0.5 * (s.age * 5.0).sin();
        draw_circle(q.x, q.y, 6.0, Color::new(1.0, 0.95, 0.6, 1.0));
        draw_circle_lines(
            q.x,
            q.y,
            10.0 + pulse * 6.0,
            2.0,
            Color::new(1.0, 0.9, 0.4, 0.8),
        );
        draw_text(
            "query",
            q.x + 12.0,
            q.y + 4.0,
            16.0,
            Color::new(1.0, 0.9, 0.5, 1.0),
        );
    }

    fn draw_hud(&self) {
        let lines = [
            "Proxima — interactive vector graph".to_string(),
            format!(
                "nodes: {}    components: {}",
                self.nodes.len(),
                self.components
            ),
            format!(
                "k: {}    metric: {}    fps: {}",
                self.k,
                if self.use_cosine { "cosine" } else { "L2" },
                get_fps()
            ),
            "P search · A add · drag move/pan".to_string(),
            "Del remove · +/- k · M metric · R reset".to_string(),
        ];
        draw_rectangle(
            12.0,
            12.0,
            420.0,
            18.0 * lines.len() as f32 + 14.0,
            Color::new(0.08, 0.09, 0.12, 0.78),
        );
        let mut y = 32.0;
        for (i, line) in lines.iter().enumerate() {
            let col = if i == 0 {
                WHITE
            } else {
                Color::new(0.78, 0.82, 0.9, 1.0)
            };
            draw_text(line, 22.0, y, if i == 0 { 20.0 } else { 17.0 }, col);
            y += 18.0;
        }
    }

    fn draw_results_panel(&self) {
        let Some(s) = &self.search else { return };
        let metric = if self.use_cosine { "cosine" } else { "L2" };
        let mut lines = vec![
            format!("brute-force search ({metric})"),
            format!(
                "scanned {} vectors, kept top {}",
                s.scanned,
                s.results.len()
            ),
        ];
        for (rank, nb) in s.results.iter().enumerate() {
            let label = self
                .nodes
                .iter()
                .find(|n| n.id == nb.id)
                .map(|n| n.label.as_str())
                .unwrap_or("?");
            lines.push(format!(
                "#{}  {}  ·  d = {:.4}",
                rank + 1,
                label,
                nb.distance
            ));
        }
        let w = 320.0;
        let h = 18.0 * lines.len() as f32 + 14.0;
        let x = screen_width() - w - 12.0;
        draw_rectangle(x, 12.0, w, h, Color::new(0.08, 0.09, 0.12, 0.82));
        let mut y = 32.0;
        for (i, line) in lines.iter().enumerate() {
            let col = if i < 2 {
                Color::new(1.0, 0.9, 0.55, 1.0)
            } else {
                Color::new(0.82, 0.86, 0.92, 1.0)
            };
            draw_text(line, x + 10.0, y, 16.0, col);
            y += 18.0;
        }
    }
}

const PALETTE: [(f32, f32, f32); 8] = [
    (0.97, 0.42, 0.42),
    (0.38, 0.65, 0.98),
    (0.72, 0.49, 0.91),
    (0.36, 0.82, 0.60),
    (0.98, 0.78, 0.32),
    (0.95, 0.55, 0.78),
    (0.45, 0.84, 0.86),
    (0.90, 0.62, 0.38),
];

fn component_color(c: usize) -> Color {
    let (r, g, b) = PALETTE[c % PALETTE.len()];
    Color::new(r, g, b, 1.0)
}

fn with_alpha(c: Color, a: f32) -> Color {
    Color::new(c.r, c.g, c.b, a)
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut app = App::new();
    loop {
        app.update();
        app.draw();
        next_frame().await;
    }
}
