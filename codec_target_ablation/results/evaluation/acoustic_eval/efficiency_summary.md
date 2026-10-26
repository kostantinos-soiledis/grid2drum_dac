# Efficiency Summary

```
                      model             device_name batch_size num_examples best_checkpoint_epoch num_parameters RTF E2E Audio Sec/Sec Model Fwd (s) Codec Decode (s) Export Wall (s) Train Wall (s) Val Wall (s) Time To Best (s) Train Steps/Sec Train Tokens/Sec Train Audio Sec/Sec Peak GPU Mem (MB) Peak GPU Res (MB) Export Peak GPU Mem (MB) Export Peak GPU Res (MB)
          postsnap_full1024 NVIDIA GeForce RTX 3080          4         1733                   147       93316608   0.065        15.357       147.008            5.285         215.104                                          0.0                                                                                                            1983.5                   4894.0
postsnap_full1024_projected NVIDIA GeForce RTX 3080          4         1733                   147       93316608   0.139         7.196       313.479           24.549         459.043                                          0.0                                                                                                            1994.6                   4894.0
              tokens_9x1024                                            1733                     7       85871104                                                                                                               0.0
```
