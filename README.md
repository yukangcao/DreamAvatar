<div align="center">

# DreamAvatar: Text-and-Shape Guided 3D Human Avatar Generation via Diffusion Models
  
<a href="https://yukangcao.github.io/">Yukang Cao</a><sup>\*</sup>,
<a href="https://yanpei.me/">Yan-Pei Cao</a><sup>\*</sup>,
<a href="https://www.kaihan.org/">Kai Han</a><sup>†</sup>,
<a href="https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=zh-CN">Ying Shan</a>,
<a href="https://i.cs.hku.hk/~kykwong/">Kwan-Yee K. Wong</a>


[![Paper](http://img.shields.io/badge/Paper-arxiv.2306.03038-B31B1B.svg)](https://arxiv.org/abs/2304.00916)
<a href="https://yukangcao.github.io/DreamAvatar/"><img alt="page" src="https://img.shields.io/badge/Webpage-0054a6?logo=Google%20chrome&logoColor=white"></a>

<img src="./docs//static/gif/clown.gif" width="200px">
<img src="./docs//static/gif/deadpool.gif" width="200px">
<img src="./docs/static/gif/joker.gif" width="200px">
<img src="./docs/static/gif/link.gif" width="200px">
  
Please refer to our webpage for more visualizations.
</div>

## Abstract
We present **DreamAvatar**, a text-and-shape guided framework for generating high-quality 3D human avatars with controllable poses. While encouraging results have been reported by recent methods on text-guided 3D common object generation, generating high-quality human avatars remains an open challenge due to the complexity of the human body's shape, pose, and appearance. We propose DreamAvatar to tackle this challenge, which utilizes a trainable NeRF for predicting density and color for 3D points and pretrained text-to-image diffusion models for providing 2D self-supervision. Specifically, we leverage the SMPL model to provide shape and pose guidance for the generation. We introduce a dual-observation-space design that involves the joint optimization of a canonical space and a posed space that are related by a learnable deformation field. This facilitates the generation of more complete textures and geometry faithful to the target pose. We also jointly optimize the losses computed from the full body and from the zoomed-in 3D head to alleviate the common multi-face ''Janus'' problem and improve facial details in the generated avatars. Extensive evaluations demonstrate that DreamAvatar significantly outperforms existing methods, establishing a new state-of-the-art for text-and-shape guided 3D human avatar generation.

<div align="center">
<img src="./docs/static/video/Pipeline-n.png" width="800px">
</div>

## Code
We will release the code soon... 🚧 Stay tuned!

## Misc.
If you want to cite our work, please use the following bib entry:
```
@article{cao2023dreamavatar,
  title={Dreamavatar: Text-and-shape guided 3d human avatar generation via diffusion models},
  author={Cao, Yukang and Cao, Yan-Pei and Han, Kai and Shan, Ying and Wong, Kwan-Yee K},
  journal={arXiv preprint arXiv:2304.00916},
  year={2023}
}
```

## Acknowledgement
Thanks to the brilliant works from [Threestudio](https://github.com/threestudio-project/threestudio) and [Stable-DreamFusion](https://github.com/ashawkey/stable-dreamfusion)
