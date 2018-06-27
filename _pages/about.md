---
permalink: /
title: "academicpages is a ready-to-fork GitHub Pages template for academic personal websites"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% include base_path %}

I am a master of research student in Web Science and Big Data Analytics at UCL. My research interest includes reinforcement learning and computer vision. I also participated in projects in natural language processing and information retrieval.

Education
======
* M.R. in Web Science and Big Data Analytics, University College London, 2017-now  
  Advisor: Prof. Jun Wang
* B.S. in Electronic Information Engineering, Beijing Institute of Technology, 2013-2017  
  Advisor: Prof. Huiqi Li
  

Publications
======
  <ul>{% for post in site.publications %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Talks
======
  <ul>{% for post in site.talks %}
    {% include archive-single-talk-cv.html %}
  {% endfor %}</ul>

