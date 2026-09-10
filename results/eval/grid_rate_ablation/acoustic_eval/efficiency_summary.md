# Efficiency Summary

```
            model             device_name batch_size num_examples best_checkpoint_epoch num_parameters RTF E2E Audio Sec/Sec Model Fwd (s) Codec Decode (s) Export Wall (s) Train Wall (s) Val Wall (s) Time To Best (s) Train Steps/Sec Train Tokens/Sec Train Audio Sec/Sec Peak GPU Mem (MB) Peak GPU Res (MB) Export Peak GPU Mem (MB) Export Peak GPU Res (MB)
grid120hz_25steps NVIDIA GeForce RTX 3080          4         1733                    72       91853384   0.058        17.249       128.880            4.880         191.511                                          0.0                                                                                                            1968.1                   4622.0
grid250hz_25steps NVIDIA GeForce RTX 3080          4         1733                   146       91853384   0.061        16.360       137.694            4.912         201.915                                          0.0                                                                                                            1968.2                   4870.0
 grid90hz_25steps NVIDIA GeForce RTX 3080          4         1733                    72       91853384   0.055        18.290       120.551            4.603         180.611                                          0.0                                                                                                            1968.0                   4708.0
```
