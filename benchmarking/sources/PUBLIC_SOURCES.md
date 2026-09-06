# benchmark reference images

The arrays in this directory are compact reference images for the registration
benchmark. Eight are derived from public imaging archives or the existing
FibreSight examples; two paired images were supplied by Zhuoyang Ye. They are
kept here so that the multi-anatomy benchmark can be regenerated without the
original recordings.

## FibreSight laboratory examples

| array | session / asset | attribution | acquisition / download | extraction | processing notes |
| --- | --- | --- | --- | --- | --- |
| `lab-fibresight-demo-train-01.npy` | training-split session; source identifier not published | released with FibreSight by Dinghao Luo; acquisition credit is not recorded in the public repository | not applicable; copied from the bundled FibreSight example, with no download tool | unlabelled 256 × 256 crop from a channel-2 axonal reference image | moved and renamed without changing the array; the full image and ROI labels remain private |
| `lab-fibresight-demo-train-02.npy` | training-split session; source identifier not published | released with FibreSight by Dinghao Luo; acquisition credit is not recorded in the public repository | not applicable; copied from the bundled FibreSight example, with no download tool | unlabelled 256 × 256 crop from a channel-2 axonal reference image | moved and renamed without changing the array; the full image and ROI labels remain private |
| `lab-fibresight-demo-test.npy` | test-split session; source identifier not published | released with FibreSight by Dinghao Luo; acquisition credit is not recorded in the public repository | not applicable; copied from the bundled FibreSight example, with no download tool | unlabelled 256 × 256 crop from a channel-2 axonal reference image | moved and renamed without changing the array; the full image and ROI labels remain private |

The public repository does not retain the original session identifiers,
acquisition software, or the position of each crop within its source image.
Those fields cannot be reconstructed from the released arrays.

## Allen Brain Observatory, Visual Coding

Download tool: AllenSDK 2.16.2. Attribution: Allen Institute Brain Observatory
Visual Coding dataset.

| array | experiment/session | Cre line | area | imaging depth | source | extraction and processing |
| --- | ---: | --- | --- | ---: | --- | --- |
| `allen-vis-two-photon-01.npy` | 503772253 | Cux2-CreERT2 | VISpm | 175 µm | [Allen NWB](https://api.brain-map.org/api/v2/well_known_file_download/516247313) | central 256 × 256 crop of the 512 × 512 motion-correction maximum-intensity projection; saved as float32 |
| `allen-vis-two-photon-02.npy` | 588535615 | Sst-IRES-Cre | VISal | 275 µm | [Allen NWB](https://api.brain-map.org/api/v2/well_known_file_download/588657253) | central 256 × 256 crop of the 512 × 512 motion-correction maximum-intensity projection; saved as float32 |

The Allen NWB files contain external links for the raw two-photon HDF5 data;
those external movie files are not downloadable through the AllenSDK catalogue
for these experiments. The saved arrays therefore use the motion-correction
reference images stored inside the public NWB files. The session metadata and
the image provenance remain Allen data; the arrays are not temporal medians of
the unavailable external movies.

## DANDI:000168, jGCaMP8

DANDI client version: 0.77.0. Attribution: contributors to
[DANDI:000168](https://dandiarchive.org/dandiset/000168).

| array | asset | cell/session | indicator | source | extraction and processing |
| --- | --- | --- | --- | --- | --- |
| `dandi-jgcamp8f-soma.npy` | `068533f7-b682-4071-8999-ca8d800f18a8` | `ANM478407_cell05` | jGCaMP8f | [DANDI asset](https://dandiarchive.s3.amazonaws.com/blobs/d40/572/d40572a9-6356-4ebb-9746-20d474b255c5) | temporal median of 300 middle frames from `acquisition/Registered movie 0/data`, central 128 × 128 crop, saved as float32 |
| `dandi-jgcamp8s-soma.npy` | `eec71d27-3ec4-40fa-a8f0-3be4d4082294` | `ANM479116_cell01` | jGCaMP8s | [DANDI asset](https://dandiarchive.s3.amazonaws.com/blobs/84c/122/84c1221e-1d39-4f00-89e1-84be8d7b5423) | temporal median of 300 middle frames from `acquisition/Registered movie 0/data`, central 128 × 128 crop, saved as float32 |

The source NWB assets are
`jGCaMP8f/jGCaMP8f_ANM478407_cell05.nwb` and
`jGCaMP8s/jGCaMP8s_ANM479116_cell01.nwb` in DANDI:000168. The images were
read with HTTP range requests through the assets' public S3 objects; the full
multi-gigabyte NWB files were not downloaded.

## DANDI:001170, olfactory bulb

DANDI client version: 0.77.0. Attribution: contributors to
[DANDI:001170](https://dandiarchive.org/dandiset/001170).

| array | asset | session | indicator/region | source | extraction and processing |
| --- | --- | --- | --- | --- | --- |
| `dandi-olfactory-bulb.npy` | `07eaf0c2-6ae8-4e7e-ad32-f5996cf74414` | `sub-124_ses-3` | olfactory bulb | [DANDI asset](https://dandiarchive.s3.amazonaws.com/blobs/4cc/c25/4ccc2514-9a72-4edb-ab1e-d999f478851c) | central 120 × 120 crop of `processing/ophys/average_images/average_image_FOV1_channel1`, edge-padded to 128 × 128 and saved as float32 |

This NWB asset contains average FOV images, fluorescence traces, and ROI
segmentation, but no frame-wise `TwoPhotonSeries` movie. The saved array is
therefore the public average FOV image, with that limitation kept explicit.

## Zhuoyang Ye, paired somatic calcium imaging

| array | session / asset | channel | attribution | acquisition / download | extraction and processing |
| --- | --- | --- | --- | --- | --- |
| `lab-session-1-channel-2.npy` | `lab-session-1` | channel 2, structural control | Zhuoyang Ye | not applicable; supplied PNG, channel 2 | further centre 256 × 256 crop of the supplied PNG, converted to float64 |
| `lab-session-1-channel-1.npy` | `lab-session-1` | channel 1, activity-dependent signal | Zhuoyang Ye | not applicable; supplied PNG, channel 1 | further centre 256 × 256 crop of the supplied PNG, converted to float64 |

These two arrays are paired channels from the same field of view. The tdTomato
image is the structural control and the GCaMP image is the signal channel; the
benchmark uses them as separate anatomical sources so that the activity-rich
signal image is tested alongside the control image.

## processing notes

The source arrays live beside this file under `benchmarking/sources/`, with a
same-stem PNG beside every array for quick visual inspection. Those PNGs are
display derivatives scaled to the 1st and 99th intensity percentiles; they are
not used by the benchmark. The public archive arrays are saved as `float32`
NumPy arrays, except for the two Zhuoyang Ye arrays, which retain `float64`
because the supplied PNGs were converted directly from their greyscale pixel
values before the additional central crop. The Allen images are cropped to the
central 256 × 256 FOV for the benchmark; the DANDI references are cropped or
padded to square FOVs because their source movies are 512 × 128 and the
olfactory average image is 502 × 120. The benchmark generator reads all ten
arrays from this directory.

## validation

The gallery was regenerated on 3 September 2026. Every array is two dimensional
and finite. The public archive arrays are `float32`; the two Zhuoyang Ye arrays
are `float64`.

| array | shape | intensity range | mean Sobel magnitude | mean Sobel after per-image 1st-99th percentile scaling |
| --- | --- | ---: | ---: | ---: |
| `allen-vis-two-photon-01.npy` | 256 × 256 | 195 to 3536 | 343.87 | 0.221450 |
| `allen-vis-two-photon-02.npy` | 256 × 256 | 244 to 2970 | 250.19 | 0.478362 |
| `dandi-jgcamp8f-soma.npy` | 128 × 128 | 1 to 775 | 146.73 | 0.239428 |
| `dandi-jgcamp8s-soma.npy` | 128 × 128 | 7.5 to 730.5 | 157.68 | 0.457796 |
| `dandi-olfactory-bulb.npy` | 128 × 128 | -5.20 to 3918.79 | 38.83 | 0.275935 |
| `lab-session-1-channel-2.npy` | 256 × 256 | 30 to 235 | 38.47 | 0.223471 |
| `lab-session-1-channel-1.npy` | 256 × 256 | 39 to 208 | 18.77 | 0.140183 |

For comparison, the three existing demo references have raw mean Sobel
magnitudes of 71.37, 84.75, and 121.30; their per-image scaled values are
0.354221, 0.359369, and 0.479216. The scaled values are the useful comparison
because the public sources retain different camera and indicator intensity
scales.
