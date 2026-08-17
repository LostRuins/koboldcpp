#include "llama-kv-cells.h"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "test-llama-kv-cells: " << message << '\n';
        std::exit(1);
    }
}

void add_cell(llama_kv_cells & cells, uint32_t i, llama_pos pos, llama_seq_id seq) {
    cells.pos_set(i, pos);
    cells.seq_add(i, seq);
}

void test_reset_and_used_range() {
    llama_kv_cells cells;
    cells.resize(6);

    require(cells.size() == 6, "resize must set the cell count");
    require(cells.get_used() == 0, "new cells must be unused");
    require(cells.used_min() == 0, "an empty cache must have used_min 0");
    require(cells.used_max_p1() == 0, "an empty cache must have used_max_p1 0");
    require(cells.seq_pos_min(0) == -1, "an absent sequence must have no minimum position");
    require(cells.seq_pos_max(0) == -1, "an absent sequence must have no maximum position");

    add_cell(cells, 4, 12, 0);
    add_cell(cells, 1, 3, 0);

    require(cells.get_used() == 2, "occupied cells must be counted");
    require(cells.used_min() == 1, "used_min must return the first occupied index");
    require(cells.used_max_p1() == 5, "used_max_p1 must be one past the last occupied index");
    require(cells.seq_pos_min(0) == 3, "sequence minimum must track occupied positions");
    require(cells.seq_pos_max(0) == 12, "sequence maximum must track occupied positions");

    cells.reset();
    require(cells.get_used() == 0, "reset must clear the used set");
    require(cells.is_empty(1) && cells.is_empty(4), "reset must empty all cells");
    require(cells.seq_pos_min(0) == -1 && cells.seq_pos_max(0) == -1,
            "reset must clear sequence position indexes");
}

void test_shared_sequences_and_duplicate_positions() {
    llama_kv_cells cells;
    cells.resize(4);

    add_cell(cells, 0, 7, 0);
    cells.seq_add(0, 1);

    require(cells.seq_count(0) == 2, "a cell must support multiple owning sequences");
    require(cells.seq_has(0, 0) && cells.seq_has(0, 1), "both sequence owners must be recorded");
    require(!cells.seq_rm(0, 0), "removing one of two owners must not empty the cell");
    require(!cells.is_empty(0) && cells.seq_get(0) == 1, "the remaining owner must be preserved");
    require(cells.seq_pos_min(0) == -1, "removed sequence positions must be updated");
    require(cells.seq_pos_min(1) == 7, "remaining sequence positions must be preserved");
    require(cells.seq_rm(0, 1), "removing the final owner must empty the cell");
    require(cells.is_empty(0) && cells.get_used() == 0, "the final removal must release the cell");

    add_cell(cells, 1, 9, 2);
    add_cell(cells, 2, 9, 2);
    require(cells.seq_pos_min(2) == 9 && cells.seq_pos_max(2) == 9,
            "duplicate positions must be represented as one range value");
    cells.rm(1);
    require(cells.seq_pos_min(2) == 9 && cells.seq_pos_max(2) == 9,
            "removing one duplicate must retain the other position occurrence");
    cells.rm(2);
    require(cells.seq_pos_min(2) == -1 && cells.seq_pos_max(2) == -1,
            "removing the final duplicate must clear the position index");
}

void test_sequence_keep() {
    llama_kv_cells cells;
    cells.resize(3);

    add_cell(cells, 0, 1, 0);
    cells.seq_add(0, 1);
    add_cell(cells, 1, 2, 1);
    add_cell(cells, 2, 3, 2);

    require(!cells.seq_keep(0, 1), "keeping an owner must retain a shared cell");
    require(cells.seq_count(0) == 1 && cells.seq_get(0) == 1,
            "seq_keep must remove all owners except the requested sequence");
    require(!cells.seq_keep(1, 1), "keeping the sole owner must retain the cell");
    require(cells.seq_keep(2, 1), "a used cell without the requested sequence must be emptied");
    require(cells.is_empty(2), "seq_keep must release cells belonging only to other sequences");
    require(cells.seq_pos_min(0) == -1, "seq_keep must remove discarded sequence indexes");
    require(cells.seq_pos_min(1) == 1 && cells.seq_pos_max(1) == 2,
            "seq_keep must preserve the kept sequence range");
}

void test_position_shifts() {
    llama_kv_cells cells;
    cells.resize(3);

    add_cell(cells, 0, 10, 0);
    add_cell(cells, 1, 3, 0);

    require(!cells.get_has_shift(), "new cells must not report pending shifts");
    require(!cells.pos_add(0, -4), "a non-negative shifted position must remain occupied");
    require(cells.pos_get(0) == 6 && cells.get_shift(0) == -4,
            "pos_add must update the position and accumulated shift");
    require(cells.seq_pos_min(0) == 3 && cells.seq_pos_max(0) == 6,
            "pos_add must update sequence position indexes");

    cells.pos_div(0, 2);
    require(cells.pos_get(0) == 3 && cells.get_shift(0) == -1,
            "pos_div must update the position and accumulated shift");
    require(cells.get_has_shift(), "position updates must set has_shift");

    cells.reset_shift();
    require(!cells.get_has_shift(), "reset_shift must clear has_shift");
    require(cells.get_shift(0) == 0 && cells.pos_get(0) == 3,
            "reset_shift must clear shifts without changing positions");

    require(cells.pos_add(1, -4), "a negative shifted position must release the cell");
    require(cells.is_empty(1) && cells.get_used() == 1,
            "a negative shift must update occupancy bookkeeping");
}

void test_copy_and_restore() {
    llama_kv_cells source;
    source.resize(5);

    add_cell(source, 1, 11, 0);
    source.seq_add(1, 2);
    source.ext_set(1, {4, 8});
    add_cell(source, 3, 13, 1);
    source.ext_set(3, {5, 9});

    const llama_kv_cells contiguous = source.cp(1, 3);
    llama_kv_cells restored;
    restored.resize(5);
    restored.set(0, contiguous);

    require(restored.get_used() == 2, "contiguous restore must rebuild the used index");
    require(restored.pos_get(0) == 11 && restored.seq_has(0, 0) && restored.seq_has(0, 2),
            "contiguous restore must preserve positions and sequence owners");
    require(restored.ext_get(0).x == 4 && restored.ext_get(0).y == 8,
            "contiguous restore must preserve extended positions");
    require(restored.is_empty(1) && restored.pos_get(2) == 13,
            "contiguous restore must preserve empty and occupied cells");
    require(restored.seq_pos_min(1) == 13 && restored.seq_pos_max(1) == 13,
            "contiguous restore must rebuild sequence position indexes");

    const std::vector<uint32_t> source_idxs = {3, 1};
    const llama_kv_cells indexed = source.cp(source_idxs);
    llama_kv_cells scattered;
    scattered.resize(6);
    const std::vector<uint32_t> destination_idxs = {4, 2};
    scattered.set(destination_idxs, indexed);

    require(scattered.pos_get(4) == 13 && scattered.seq_has(4, 1),
            "indexed restore must map the first source cell to the first destination");
    require(scattered.pos_get(2) == 11 && scattered.seq_has(2, 0) && scattered.seq_has(2, 2),
            "indexed restore must preserve shared ownership");
    require(scattered.used_min() == 2 && scattered.used_max_p1() == 5,
            "indexed restore must rebuild used bounds");
}

void test_extended_positions() {
    llama_kv_cell_ext ext = {2, 3};

    require(ext.is_2d_gt(1, 3), "x must break ties when y is equal");
    require(ext.is_2d_gt(100, 2), "a greater y must compare greater regardless of x");
    require(!ext.is_2d_gt(2, 3), "equal 2D positions must not compare greater");
    require(!ext.is_2d_gt(3, 3), "a smaller x at equal y must not compare greater");

    ext.reset();
    require(ext.x == 0 && ext.y == 0, "extended position reset must clear both coordinates");
}

} // namespace

int main() {
    test_reset_and_used_range();
    test_shared_sequences_and_duplicate_positions();
    test_sequence_keep();
    test_position_shifts();
    test_copy_and_restore();
    test_extended_positions();

    std::cout << "test-llama-kv-cells: all tests passed\n";
    return 0;
}
