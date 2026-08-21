mod support;

use oneliner::model;
use oneliner::runtime::ModelInference;

#[model(
    "../../examples/models/vmcu_pointwise_pair.mlir",
    backend = "iree",
    arena = "owned"
)]
struct PointwisePairBaselineOwned;

const INPUT: [i8; 16] = [1, 2, 3, 4, -1, 0, 1, 2, 5, -3, 2, 1, 127, -128, 64, -64];
const EXPECTED: [i8; 12] = [8, 17, 10, 2, 1, 6, -18, 21, 12, -128, -1, 0];

#[test]
fn pointwise_pair_baseline_runs_with_owned_arena() {
    support::assert_artifacts::<PointwisePairBaselineOwned>("pointwise pair baseline (owned)");

    let mut model = PointwisePairBaselineOwned::new();
    let mut input = PointwisePairBaselineOwned::create_input_tensor();
    input.as_slice_mut().copy_from_slice(&INPUT);

    let output = model.run(&input);
    assert_eq!(output.as_slice(), EXPECTED);
}
