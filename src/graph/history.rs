//! Undo / redo for the node graphs (material, world and processing together, so one
//! Ctrl+Z history covers every editor mode). Graphs are small, so each step is a full
//! snapshot, taken whenever an edit has settled (no drag or text entry in progress): a
//! whole slider drag or node move becomes a single step.

use super::{Graph, Graphs};

const LIMIT: usize = 200;

#[derive(Default)]
pub struct History {
    undo: Vec<Graphs>,
    redo: Vec<Graphs>,
    /// The graphs as of the last commit.
    current: Option<Graphs>,
}

impl History {
    /// Records `g` as a new step if it changed since the last commit.
    pub fn commit(&mut self, g: &Graphs) {
        if self.current.as_ref().is_some_and(|c| c.same_content(g)) {
            return;
        }
        if let Some(prev) = self.current.replace(g.clone()) {
            self.undo.push(prev);
            if self.undo.len() > LIMIT {
                self.undo.remove(0);
            }
            self.redo.clear();
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo(&mut self, g: &mut Graphs) -> bool {
        self.commit(g);
        let Some(prev) = self.undo.pop() else { return false };
        self.redo.push(self.current.replace(prev.clone()).expect("committed above"));
        restore_all(g, prev);
        true
    }

    pub fn redo(&mut self, g: &mut Graphs) -> bool {
        // An edit since the last undo commits here and clears the redo stack.
        self.commit(g);
        let Some(next) = self.redo.pop() else { return false };
        self.undo.push(self.current.replace(next.clone()).expect("committed above"));
        restore_all(g, next);
        true
    }
}

fn restore_all(g: &mut Graphs, snapshot: Graphs) {
    let Graphs { material, world, post } = snapshot;
    restore(&mut g.material, material);
    restore(&mut g.world, world);
    restore(&mut g.post, post);
}

/// Swaps in a snapshot, bumping the revision only when the shader has to change, so
/// undoing a slider tweak doesn't trigger a recompile.
fn restore(g: &mut Graph, snapshot: Graph) {
    let rev = g.structure_rev + !g.same_structure(&snapshot) as u64;
    *g = snapshot;
    g.structure_rev = rev;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{NodeKind, preset};

    #[test]
    fn undo_redo_round_trip() {
        let mut gs = Graphs { material: preset("Mandelbulb").0, ..Graphs::default() };
        let mut h = History::default();
        h.commit(&gs);
        let original = gs.clone();

        // A value edit: undoing it must not force a recompile.
        let id = gs.material.nodes.iter().find(|n| n.kind == NodeKind::Mandelbulb).unwrap().id;
        let v = gs.material.node(id).unwrap().values[0];
        gs.material.node_mut(id).unwrap().values[0] = v + 1.0;
        h.commit(&gs);
        // A structural edit.
        gs.material.add(NodeKind::Time, [0.0, 0.0]);
        h.commit(&gs);
        // An edit in another graph is a step of the same history.
        gs.world.add(NodeKind::StarField, [0.0, 0.0]);
        h.commit(&gs);
        let edited = gs.clone();

        assert!(h.undo(&mut gs));
        assert!(!gs.world.nodes.iter().any(|n| n.kind == NodeKind::StarField));
        let rev = gs.material.structure_rev;
        assert!(h.undo(&mut gs));
        assert!(gs.material.structure_rev > rev, "removing a node needs a recompile");
        let rev = gs.material.structure_rev;
        assert!(h.undo(&mut gs));
        assert_eq!(gs.material.structure_rev, rev, "value-only undo must not recompile");
        assert!(gs.same_content(&original));
        assert!(!h.undo(&mut gs));

        assert!(h.redo(&mut gs) && h.redo(&mut gs) && h.redo(&mut gs));
        assert!(gs.same_content(&edited));
        assert!(!h.redo(&mut gs));

        // A new edit after undo drops the redo stack.
        h.undo(&mut gs);
        gs.material.node_mut(id).unwrap().values[0] = v - 1.0;
        assert!(!h.redo(&mut gs));
    }
}
