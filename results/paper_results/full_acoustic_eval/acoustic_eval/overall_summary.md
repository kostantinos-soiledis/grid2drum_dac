# Overall Summary

```
                       model num_examples  FAD∞ FAD∞ R² Mel MAE (dB) Broad Flux Cos Band Balance L1 Centroid MAE (Hz) RMS MAE (dB) Crest MAE (dB) RTF E2E Audio Sec/Sec Train Wall (s) Time To Best (s) Peak GPU Mem (MB)
            target_pca_recon         1733 0.016   0.847        0.096          0.999           0.000              11.1        7.288          0.008   0.015        67.095                             0.0                  
          source_code_decode         1733 0.016   0.848        0.096          0.999           0.000              11.1        7.288          0.008   0.015        66.580                             0.0                  
            target_dac_recon         1733 0.016   0.848        0.096          0.999           0.000              11.1        7.288          0.008   0.015        66.100                             0.0                  
       diffusion_pca_25steps         1733 0.019   0.891        5.686          0.848           0.034             394.0        6.964          1.606   0.077        13.040                             0.0                  
diffusion_pca_rvq_ce_25steps         1733 0.020   0.848        5.468          0.863           0.033             364.6        6.422          1.716   0.072        13.871                             0.0                  
diffusion_pca_rvq_ce_12steps         1733 0.020   0.845        5.394          0.864           0.033             352.8        6.754          1.595   0.046        21.906                             0.0                  
       diffusion_pca_12steps         1733 0.021   0.835        5.754          0.850           0.035             408.8        6.758          1.626   0.065        15.483                             0.0                  
 diffusion_pca_rvq_ce_6steps         1733 0.022   0.830        5.470          0.866           0.040             408.3        6.770          1.613   0.027        36.456                             0.0                  
        diffusion_pca_6steps         1733 0.023   0.851        6.387          0.843           0.042             546.5        6.749          1.696   0.060        16.773                             0.0                  
       diffusion_pca_50steps         1733 0.024   0.744        5.706          0.839           0.039             397.9        6.604          1.729   0.107         9.322                             0.0                  
           symbolic_nn_train         1733 0.025   0.846       17.544          0.330           0.100            1122.0        7.646          2.392   0.018        55.415                             0.0                  
direct_pca_d1024_l6_seed1234         1733 0.355   0.169       13.036          0.836           0.169            1723.1        5.532          8.309   0.021        46.581                             0.0                  
                 grid_render         1733 0.551   0.147       19.267          0.763           0.082            2119.0        9.826          2.809   0.004       284.952                             0.0                  
```
