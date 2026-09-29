# ResShift seed experiment

30 test images (10 per content) x 2 JPEG levels x 5 seeds. Metric columns are the across-seed standard deviation, averaged over images. `match_flip` = fraction of faces whose match/no-match decision differs between seeds; `cer_range` = max - min CER across seeds.

| content   | jpeg   |   pixel_std |   lpips_between_seeds |   psnr |   ssim |   lpips |   musiq |   face_sim |      cer |   match_flip |   cer_range |
|:----------|:-------|------------:|----------------------:|-------:|-------:|--------:|--------:|-----------:|---------:|-------------:|------------:|
| faces     | none   |      2.5858 |                0.0504 | 0.2643 | 0.0074 |  0.0067 |  0.7304 |     0.003  | nan      |            0 |    nan      |
| faces     | q30    |      2.8833 |                0.0873 | 0.2412 | 0.0066 |  0.0093 |  3.2295 |     0.0156 | nan      |            0 |    nan      |
| objects   | none   |      5.0653 |                0.0743 | 0.106  | 0.004  |  0.0059 |  0.9895 |   nan      | nan      |          nan |    nan      |
| objects   | q30    |      6.3685 |                0.1519 | 0.1298 | 0.0069 |  0.0084 |  1.7112 |   nan      | nan      |          nan |    nan      |
| text      | none   |      5.5742 |                0.0425 | 0.1272 | 0.0022 |  0.0028 |  0.4409 |   nan      |   0.0111 |          nan |      0.0271 |
| text      | q30    |      7.0904 |                0.062  | 0.1818 | 0.004  |  0.0042 |  0.6695 |   nan      |   0.0117 |          nan |      0.0289 |
