# Suggested completion commands

Run these only after checking GPU availability. They are not executed by build_paper_results.py.

## Canonical full acoustic evaluation
python scripts/run_diffusion_acoustic_eval.py \
  --prediction-dirs <DRUMTOGRID_ROOT>/runs_baselines/dac_test_v1/target_dac_recon \
    <DRUMTOGRID_ROOT>/runs_baselines/dac_test_v1/target_pca_recon \
    <DRUMTOGRID_ROOT>/runs_baselines/dac_test_v1/grid_render \
    <DRUMTOGRID_ROOT>/runs_baselines/dac_test_v1/source_code_decode \
    <DRUMTOGRID_ROOT>/runs_baselines/dac_test_v1/symbolic_nn_train \
    <DRUMTOGRID_ROOT>/runs_direct/direct_pca_d1024_l6_seed1234/test_set_predictions \
  --prediction-names target_dac_recon target_pca_recon grid_render source_code_decode symbolic_nn_train direct_pca_d1024_l6_seed1234 \
  --cache-root cache_4beats_dac44q9_pca72_native_bpmgeom_duration_v1 \
  --out-dir paper_results/full_acoustic_eval \
  --fad-model clap-laion-music \
  --fad-workers 1 --fad-inf-workers 1 \
  --batch-size 16 --num-workers 4 --device auto \
  --with-inference --no-plots --overwrite

## Faster fallback if paired inference is too slow
python scripts/run_diffusion_acoustic_eval.py \
  --prediction-dirs <DRUMTOGRID_ROOT>/runs_baselines/dac_test_v1/target_dac_recon \
    <DRUMTOGRID_ROOT>/runs_baselines/dac_test_v1/target_pca_recon \
    <DRUMTOGRID_ROOT>/runs_baselines/dac_test_v1/grid_render \
    <DRUMTOGRID_ROOT>/runs_baselines/dac_test_v1/source_code_decode \
    <DRUMTOGRID_ROOT>/runs_baselines/dac_test_v1/symbolic_nn_train \
    <DRUMTOGRID_ROOT>/runs_direct/direct_pca_d1024_l6_seed1234/test_set_predictions \
  --prediction-names target_dac_recon target_pca_recon grid_render source_code_decode symbolic_nn_train direct_pca_d1024_l6_seed1234 \
  --cache-root cache_4beats_dac44q9_pca72_native_bpmgeom_duration_v1 \
  --out-dir paper_results/full_acoustic_eval \
  --fad-model clap-laion-music \
  --fad-workers 1 --fad-inf-workers 1 \
  --batch-size 16 --num-workers 4 --device auto \
  --no-plots --overwrite

## Rebuild paper tables after all evaluations finish
python scripts/build_paper_results.py --out-dir paper_results --strict
