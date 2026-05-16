use std::time::{Duration, Instant};
use std::vec::Vec;

use crate::state::{State, checkmate_checker};
use crate::mcts::{MCTS, TIME_LIMIT_MS};
use crate::node::Node;

// + 10 * usize is the best, considering the size of children vec
const NODE_SIZE_BYTES: usize = std::mem::size_of::<Node>() + std::mem::size_of::<usize>() * 10;
const MAX_MCTS_BYTES: usize = 1000 * 1024 * 1024; // 1000 MB
const MAX_MCTS_NODES: usize = MAX_MCTS_BYTES / NODE_SIZE_BYTES;

static mut MCTS_CACHE: Option<MCTS> = None;

/// Convert Vec<Vec<i32>> to bitboards
/// 
/// Least significant bit is the bottom row!
fn convert_to_bitboards(
    board_vv: &Vec<Vec<u8>>, 
    row_count: u8, 
    col_count: u8
) -> (Vec<u64>, Vec<u64>) {
    let mut p1_cols = vec![0u64; col_count as usize]; // Opponent (1)
    let mut p2_cols = vec![0u64; col_count as usize]; // Self (2)

    for c in 0..col_count {
        for r_top in 0..row_count { // r_top is row index from top
            let piece = board_vv[r_top as usize][c as usize];
            if piece != State::Empty as u8 {
                let r_bottom = (row_count - 1) - r_top; // Convert to row index from bottom
                if piece == State::OpponentPiece as u8 {
                    p1_cols[c as usize] |= 1u64 << r_bottom;
                } else if piece == State::SelfPiece as u8 {
                    p2_cols[c as usize] |= 1u64 << r_bottom;
                }
            }
        }
    }
    (p1_cols, p2_cols)
}

#[allow(static_mut_refs)]
pub fn get_point(
    row: usize,
    col: usize,
    top: &Vec<i32>,
    board: &Vec<Vec<i32>>,
    last_x: i32,
    last_y: i32,
    nox: usize,
    noy: usize,
) -> (i32, i32) {
    for i in 0..col {
        for j in 0..row {
            if board[j][i] == State::SelfPiece as i32 {
                let (x, y) = checkmate_checker(row, col, board, j as i32, i as i32, nox, noy, board[j][i].into());
                if (x, y) != (-1, -1) {
                    return (x, y);
                }
            }
        }
    }

    for i in 0..col {
        for j in 0..row {
            if board[j][i] == State::OpponentPiece as i32 {
                let (x, y) = checkmate_checker(row, col, board, j as i32, i as i32, nox, noy, board[j][i].into());
                if (x, y) != (-1, -1) {
                    return (x, y);
                }
            }
        }
    }

    // --- Type casting ---
    let nox = nox as u8;
    let noy = noy as u8;
    let row = row as u8;
    let col = col as u8;
    let top = top.iter().map(|&x| x as u8).collect::<Vec<u8>>();
    let board = board.iter()
        .map(|v| v.iter()
                .map(|&x| x as u8)
                .collect::<Vec<u8>>())
            .collect::<Vec<Vec<u8>>>();
    // --- Type casting ---

    let start_time = Instant::now();

    let mcts_cache: &mut Option<MCTS> = unsafe { &mut MCTS_CACHE };
    
    let mut reset_needed = false;

    let (current_game_p1_cols, current_game_p2_cols) = convert_to_bitboards(&board, row, col);

    if let Some(cached_mcts) = mcts_cache.as_mut() {
        if cached_mcts.nodes.len() >= MAX_MCTS_NODES {
            reset_needed = true;
        } else if last_x != -1 && last_y != -1 { 
            if !cached_mcts.move_root_for_opponent(last_x as u8, last_y as u8) {
                reset_needed = true;
            } else {
                if cached_mcts.root_p1_cols != current_game_p1_cols ||
                   cached_mcts.root_p2_cols != current_game_p2_cols ||
                   cached_mcts.root_top != *top ||
                   cached_mcts.nodes[cached_mcts.root_idx].player != State::SelfPiece {
                    reset_needed = true;
                }
            }
        } else { 
            if cached_mcts.root_p1_cols != current_game_p1_cols ||
               cached_mcts.root_p2_cols != current_game_p2_cols ||
               cached_mcts.root_top != *top ||
               cached_mcts.nodes[cached_mcts.root_idx].player != State::SelfPiece {
                reset_needed = true;
            }
        }
    } else { 
        reset_needed = true;
    }

    if reset_needed {
        *mcts_cache = Some(MCTS::new_from_state(
            current_game_p1_cols.clone(),
            current_game_p2_cols.clone(),
            top.clone(),
            row, col, nox, noy,
            State::SelfPiece,
        ));
    }

    let mcts = mcts_cache.as_mut().unwrap();

    // MCTS search loop
    while start_time.elapsed() < Duration::from_millis(TIME_LIMIT_MS) {
        let (leaf_idx, path_to_leaf, leaf_p1_cols, leaf_p2_cols, leaf_top) =
            mcts.select_node(
                mcts.root_idx, 
                &mcts.root_p1_cols, 
                &mcts.root_p2_cols,
                &mcts.root_top,
            );

        let mut node_to_simulate_idx = leaf_idx;
        let mut p1_cols_for_sim = leaf_p1_cols.clone();
        let mut p2_cols_for_sim = leaf_p2_cols.clone();
        let mut top_for_sim = leaf_top.clone();

        if !mcts.nodes[leaf_idx].is_terminal {
            if mcts.nodes[leaf_idx].children.is_empty() { 
                 if let Some(expanded_child_idx) = mcts.expand_node(
                     leaf_idx, 
                     &leaf_p1_cols, &leaf_p2_cols, 
                     &leaf_top
                    ) {
                    node_to_simulate_idx = expanded_child_idx;
                    
                    let child_node_details = &mcts.nodes[expanded_child_idx];
                    if let Some(move_col) = child_node_details.move_col {
                        let player_who_made_move = mcts.nodes[leaf_idx].player;
                        if top_for_sim[move_col as usize] > 0 { 
                            let land_row_from_top = top_for_sim[move_col as usize] - 1;
                            let land_row_bottom = (row - 1) - land_row_from_top;
                            
                            if player_who_made_move == State::SelfPiece {
                                p2_cols_for_sim[move_col as usize] |= 1u64 << land_row_bottom;
                            } else {
                                p1_cols_for_sim[move_col as usize] |= 1u64 << land_row_bottom;
                            }
                            top_for_sim[move_col as usize] -= 1;
                        }
                    }
                 }
            }
        }
        
        let sim_result = mcts.simulate(node_to_simulate_idx, p1_cols_for_sim, p2_cols_for_sim, top_for_sim);
        mcts.backpropagate(&path_to_leaf, sim_result);
    }

    let ret: (i32, i32);

    if let Some(best_col) = mcts.get_best_move_col() {
        if top[best_col as usize] > 0 { 
            ret = ((top[best_col as usize] - 1) as i32, best_col as i32);
            mcts.move_root_for_self(best_col);
        } else {
            *mcts_cache = None; 
            ret = find_fallback_move(col, &top, nox, noy);
        }
    } else {
        // MCTS provided no best move. Reset.
        *mcts_cache = None;
        ret = find_fallback_move(col, &top, nox, noy);
    }
    
    ret
}

fn find_fallback_move(
    col: u8,
    top: &Vec<u8>,
    nox: u8,
    noy: u8,
) -> (i32, i32) {
    let mut fallback_col_idx = 0;
    let mut found_fallback = false;
    for c_idx in 0..col {
        if top[c_idx as usize] > 0 && !((top[c_idx as usize] - 1) == nox && c_idx == noy) {
            fallback_col_idx = c_idx;
            found_fallback = true;
            break;
        }
    }
    if found_fallback {
        ((top[fallback_col_idx as usize] - 1) as i32, fallback_col_idx as i32)
    } else {
        ((top.get(0).cloned().unwrap_or(0) - 1) as i32, 0)
    }
}
