<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<!-- <br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</div> -->





<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#1-about-the-project">About The Project</a>
      <ul>
        <li><a href="#11-action">Action</a></li>
        <li><a href="#how-to-find-motifs">How to find motifs</a></li>
      </ul>
    </li>
    <li>
      <a href="#dataset">Dataset</a>
      <ul>
        <li><a href="#extract-hard-samples">Extract hard samples</a></li>
        <li><a href="#preprocessing">Preprocessing</a></li>
      </ul>
    </li>
    <li><a href="#training">Training</a></li>
    <li><a href="#evaluation">Evaluation</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## 1. About The Project
MotifRetro uses motif actions to shorten the edit path for retrosynthesis prediction. The newly added motif actions inlude:
* AddMotif: Adding motif to the molecule, where there must be one bond connecting the motif and the molecule.
* ReplaceMotif: Replacing an atom or a bond as a motif.
* DelMotif: Removing incomplete sub-motif structure for adding a complete motif better.

We modify the motif discovery algorithm of PS-VAE to search candidate motif given the source and target molecules, where the innovations include:
* Keep Ring: We force PS-VAE to use the ring as a basic element, so it no longer breaks the ring.
* Dynamic search: As editing proceeds, the source molecule will change. Candidate motifs that can be added to the source must be obtained by comparing the latest source molecule with a fixed target molecule.

 

### 1.1. Action

We provide visual examples:

### 1.2. How to find motifs




## 2. Dataset

### 2.1. Extract hard samples
1. Run the jupyter: 
    ```
    ./dataset_code/extract_hard_examples.ipynb
    ```
2. Compare the action paths of MotifRetro and MEGAN:
    ```
    ./dataset_code/visualize_action_path.ipynb
    ```

### 2.2. Preprocess

#### 2.2.1. Generate edit path
- **Check the edit path generation algorithm**
  ```
    ./src/feat/test_featureizer.py
  ```

<figure class="half" style="display:flex">
<table>
  <tr>
    <td>
    <img 
      src="./assets/action_path/megan0.png" 
      alt="Alt text" 
      title="MEGAN edit path"
      style="display: inline-block; margin: 0 auto; max-width: 500px;">
      <figcaption>Edit path of MEGAN.</figcaption>
    </td>
    <td>
    <img 
      src="./assets/action_path/our0.png" 
      alt="Alt text" 
      title="MEGAN edit path"
      style="display: inline-block; margin: 0 auto; max-width: 500px;">
      <figcaption>Edit path of MotifRetro.</figcaption>
    </td>
  </tr>
</table>

</figure>

#### 2.2.2. Data format

<table style="margin-left: auto;margin-right: auto;">
<thead>
  <tr>
    <th>Name</th>
    <th>Description</th>
    <th>Shape</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="3" style="text-align: center;">featurize_parallel()</td>
  </tr>
  <tr>
    <td>data_len</td>
    <td>the number of reactions</td>
    <td></td>
  </tr>
  <tr>
    <td>samples_len</td>
    <td>the number of actions</td>
    <td>samples_len = data_len * self.max_n_steps</td>
  </tr>
  <tr>
    <td>nodes_mat</td>
    <td>node feature matrix</td>
    <td>(samples_len, max_n_nodes)</td>
  </tr>
  <tr>
    <td>adj_mat</td>
    <td>edge feature matrix</td>
    <td>(samples_len, max_n_nodes**2)</td>
  </tr>
  <tr>
    <td colspan="3" style="text-align: center;" >get_sparse_data()</td>
  </tr>
  <tr>
    <td>action_tuples</td>
    <td>list of local actions</td>
    <td></td>
  </tr>
  <tr>
    <td>sample_data</td>
    <td>local action label matrix, including 4 feature:<br>[atom_map1, atom_map2, len(nodes), is_hard]</td>
    <td>(#action, #feature)</td>
  </tr>
  <tr>
    <td>nodes_vals</td>
    <td>local node feature matrix</td>
    <td>(#action, #nodes)</td>
  </tr>
  <tr>
    <td>adj_vals</td>
    <td>local edge feature matrix</td>
    <td>(#action, #edge)</td>
  </tr>
</tbody>
</table>


<table style="margin-left: auto;margin-right: auto;">
<thead>
  <tr>
    <th>Action</th>
    <th>Description</th>
    <th>Format</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>AddAtom</td>
    <td></td>
    <td>("add_atom", (bond_feat_tuple, atom_feat_tuple))</td>
  </tr>
  <tr>
    <td>AddBond</td>
    <td></td>
    <td>("change_bond",bond_feat_tuple)</td>
  </tr>
  <tr>
    <td>AddMotif</td>
    <td></td>
    <td>("add_motif", smiles)</td>
  </tr>
  <tr>
    <td>DelBond</td>
    <td></td>
    <td>("change_bond", (None,None))</td>
  </tr>
  <tr>
    <td>DelMotif</td>
    <td></td>
    <td>("delete_motif", None)</td>
  </tr>
  <tr>
    <td>EditAtom</td>
    <td></td>
    <td>("change_atom", atom_feat_tuple)</td>
  </tr>
  <tr>
    <td>EditBond</td>
    <td></td>
    <td>("change_bond",bond_feat_tuple)</td>
  </tr>
  <tr>
    <td>ReplaceMotif</td>
    <td></td>
    <td>("replace_motif", smiles) (需要标注replace的位置)</td>
  </tr>
  <tr>
    <td>Stop</td>
    <td></td>
    <td>("stop",)</td>
  </tr>
</tbody>
</table>

### Make your data from scrach
We present the process of making training data, taking USPTOHard as an example:

  - Copy *USPTOHard.csv* and *default_split.csv* to */data/uspto_hrad*.
  - Generate Motif vocabulary by:
  ```
    cd /gaozhangyang/experiments/MotifRetro/dataset_code
    sh ps_extract.sh
  ```
  - Preprocess dataset and train the model:
  ```
    python main.py
  ```










<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 