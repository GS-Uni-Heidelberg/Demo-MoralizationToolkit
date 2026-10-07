# Moralization Toolkit

The Moralization Toolkit is a web application for exploring and
comparing moralization predictions in German, English, and French text. It
combines fine-tuned multilingual classifiers with optional OpenAI and Claude
models, and can also highlight DiMi lemma matches and return model-specific
annotations such as moral values and protagonists.

The deployed toolkit is available here: [https://moralization-toolkit.chai-lab.de/](https://moralization-toolkit.chai-lab.de/)

## Development

For local setup, deployment, backend and frontend configuration, model hosting,
and billing administration, see [DEVELOPMENT.md](DEVELOPMENT.md).

The backend provides the FastAPI API and model integrations. The frontend is a
Next.js application with the interactive analysis interface and batch-upload
workflow.

## Related Work
The Moralization Toolkit builds upon several of our previous works on moralization:

[1] Maria Becker, Bruno Brocai, and Lars Tapken. 2023. Detection and Analysis of Moralization Practices Across Languages and Domains. In Book of Abstracts, page 147.

[2] Maria Becker, Mirko Sommer, Lars Tapken, Yi Wan Teh, and Bruno Brocai. 2026. The Moralization Corpus: Frame-Based Annotation and Analysis of Moralizing Speech Acts across Diverse Text Genres. In Proceedings of the Fifteenth Language Resources and Evaluation Conference, pages 7069–7091, Palma de Mallorca, Spain. ELRA Language Resource Association.

[3] Mirko Sommer and Maria Becker. 2026. Who Plays Which Role? Protagonist Detection and Classification in Moral Discourse. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 4: Student Research Workshop), pages 375–392, Rabat, Morocco. Association for Computational Linguistics.

## Citation
If you use the software, cite the repository and our other works on moralization:

```bibtex
@software{moralization_toolkit,
	author  = {Sommer, Mirko and Becker, Maria},
  title   = {Moralization Toolkit},
	year    = {2026},
	url     = {https://github.com/GS-Uni-Heidelberg/Demo-MoralizationDetectionWeb},
	note    = {Web application: https://moralization-toolkit.chai-lab.de/}
}
```

```bibtex
@inproceedings{becker-etal-2026-moralization,
    title = "The Moralization Corpus: Frame-Based Annotation and Analysis of Moralizing Speech Acts across Diverse Text Genres",
    author = "Becker, Maria  and
      Sommer, Mirko  and
      Tapken, Lars  and
      Teh, Yi Wan  and
      Brocai, Bruno",
    editor = "Piperidis, Stelios  and
      Bel, N{\'u}ria  and
      van den Heuvel, Henk  and
      Ide, Nancy  and
      Krek, Simon  and
      Toral, Antonio",
    booktitle = "Proceedings of the Fifteenth Language Resources and Evaluation Conference",
    month = may,
    year = "2026",
    address = "Palma de Mallorca, Spain",
    publisher = "ELRA Language Resource Association",
    url = "https://aclanthology.org/2026.lrec-1.563/",
    doi = "10.63317/28h9saps9vhr",
    pages = "7069--7091"
}
```