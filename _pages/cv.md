---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
  - /cv-json/
  - /resume-json
---

{% capture cv_url %}{{ site.cv_pdf | relative_url }}?v={{ site.cv_updated | remove: '-' }}{% endcapture %}

My full, up-to-date curriculum vitae is available as a PDF:

<a href="{{ cv_url }}" class="btn btn--primary" target="_blank" rel="noopener">📄&nbsp; Download CV (PDF)</a>

Last updated: {{ site.cv_updated | date: "%B %-d, %Y" }}.

<iframe src="{{ cv_url }}" title="Curriculum vitae of Tran Dinh Phu" width="100%" height="900" style="border: 1px solid #ccc; margin-top: 1.5em;">
  This browser does not support embedded PDFs. <a href="{{ cv_url }}">Download the CV here</a>.
</iframe>
