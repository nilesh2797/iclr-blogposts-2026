---
layout: distill
title: Trade-offs in LLM Compute for Reasoning-Intensive Information Retrieval

description: This is some random description
date: 2026-04-27
future: true
htmlwidgets: true

# anonymize when submitting
authors:
  - name: Anonymous

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2026-04-27-trade-offs-in-llm-compute-for-reasoning-intensive-information-retrieval.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Background
    subsections:
      - name: Reasoning-Intensive Retrieval and BRIGHT
      - name: The Compute Levers in Neural IR
      - name: The "Thinking" Dimension
  - name: Experimental Setup
    subsections:
      - name: Model Suite
      - name: Evaluation Metrics
      - name: Experiment Design
        subsections:
          - name: Scaling Compute in Query Expansion (QE)
          - name: Scaling Compute in Reranking (RR)
          - name: Increasing Search Depth (Top-k)
          - name: Impact of "Thinking" in Reranking
          - name: Impact of Model Strength in Reranking
---

## Introduction
The paradigm of Information Retrieval (IR) is undergoing a fundamental shift. While traditional IR focused on semantic similarity and keyword matching, modern applications increasingly demand Reasoning-Intensive Information Retrieval (RIIR). In these scenarios—exemplified by the recent ICLR 2025 Spotlight paper BRIGHT—a system cannot simply find a document that looks like the query; it must understand complex logic, synthesize constraints, and deduce relationships to identify the correct evidence.

The BRIGHT benchmark established that standard retrieval methods (both sparse and dense) struggle significantly with these tasks. However, it also highlighted a promising path forward: the integration of Large Language Models (LLMs) into the retrieval pipeline. Specifically, the paper demonstrated that Query Expansion (QE) and LLM-based Reranking (RR) are critical for boosting performance.

Yet, this introduces a complex resource allocation problem. If we view "reasoning" as a finite resource (quantifiable by inference cost or latency), where is it best spent? Should we use a stronger, more expensive model to formulate a better search query (QE), or should we reserve that compute to scrutinize the retrieved documents (RR)? Furthermore, with the advent of "thinking" models (models capable of dynamic chain-of-thought generation during inference), we now have a third lever: the depth of reasoning per token.

In this blogpost, we perform a controlled ablation study to map the Pareto frontier of LLM compute in RIIR. Using the Gemini 2.5 family of models, we systematically evaluate the cost-performance trade-offs across three dimensions:

Model Strength: From lightweight models (Flash-Lite) to reasoning-heavy models (Pro).
Inference Depth: Comparing standard inference against dynamic "thinking" modes.
Search Depth: Analyzing the impact of increasing the reranking pool size (k).
Our goal is to answer a practical question for system designers: In a reasoning-intensive pipeline, where does an additional unit of compute yield the highest marginal gain in retrieval accuracy?

## Background
### Reasoning-Intensive Retrieval and BRIGHT
Standard retrieval benchmarks (e.g., BEIR) often rely on lexical overlap or semantic proximity. In contrast, the BRIGHT benchmark consists of queries where the answer requires multi-hop reasoning or domain-specific logic that is not explicitly present in the query text. For example, a query might ask about a specific chemical property that implies a class of materials, requiring the retriever to identify documents discussing those materials without the explicit class name being present.

Mathematically, let q be a complex query and \mathcal{C} be a corpus. We seek a document d^* \in \mathcal{C} that satisfies a reasoning function f(q, d) \to \{0, 1\}. Traditional retrievers approximate f via similarity scores s(q, d), which often fail when f is non-linear or logic-heavy.

### The Compute Levers in Neural IR
To bridge the gap between similarity and reasoning, we inject LLM compute at two distinct stages:

1. Query Expansion (QE): The LLM transforms the raw user query q into an enriched query q_{exp}. This is a generative process where the model hallucinates relevant context, deduces implicit constraints, or breaks down the problem.

The retrieval is then performed using a standard algorithm (e.g., BM25) on q_{exp}.

2. Reranking (RR): Given an initial retrieved list \mathcal{L} = \{d_1, ..., d_k\}, the LLM scores each document (or pairs/lists of documents) based on relevance to q.

This stage allows for deep scrutiny of the content but scales linearly (or quadratically) with the list size k.

### The "Thinking" Dimension
Recent advancements in model architectures (such as the Gemini 2.5 series) introduce the concept of inference-time compute scaling. Unlike standard models that output tokens immediately, "thinking" models generate internal chain-of-thought traces to plan and verify their reasoning before outputting a final answer. In the context of BRIGHT, this allows the model to "think" about the query's constraints during QE or "think" about the logical connection between a document and a query during RR.

### Experimental Setup
To isolate the impact of compute allocation, we fix our retrieval algorithm to BM25 (using the Pyserini implementation) and vary the LLM components used for Query Expansion and Reranking.

### Model Suite
We utilize the Google Gemini 2.5 family to represent a spectrum of cost and capability. We categorize them as follows:

Gemini-2.5-Flash-Lite: A highly efficient, low-latency model (No-Thinking mode only).
Gemini-2.5-Flash (No-Think): A standard mid-sized model (Thinking features disabled).
Gemini-2.5-Flash (Think): The same mid-sized model with dynamic thinking enabled, allowing for extended reasoning tokens.
Gemini-2.5-Pro: A large, high-capacity model (Thinking mode enabled).
### Evaluation Metrics
Quality: We report NDCG@10 for ranking quality and Recall@100 to assess the retrieval ceiling.
Performance: We measure Cost per Sample ($) and Time per Sample (ms) to visualize the trade-offs.
### Experiment Design
We structure our analysis into two primary phases:

#### Scaling Compute in Query Expansion (QE)
We first evaluate the impact of the generator's strength on the initial retrieval stage.

Protocol: For every query q in the BRIGHT subsets, we generate q_{exp} using the four model variants (Flash-Lite, Flash-No-Think, Flash-Think, Pro).
Retrieval: We perform BM25 retrieval using q_{exp}.
Objective: To determine if "smarter" queries (generated by more expensive models) lead to higher Recall@100, providing a better candidate set for the reranker.
#### Scaling Compute in Reranking (RR)
We investigate three specific mechanisms for increasing compute during the reranking stage. For these experiments, we take the retrieval outputs from Phase 1 and apply varying reranking strategies.

#### Increasing Search Depth (Top-k)

Hypothesis: Reranking more documents improves recall at the top of the list but increases cost linearly.
Setup:
Retrieval Basis: We use the candidate lists generated by all four QE settings from Phase 1.
Reranker: Fixed to Gemini-2.5-Flash (Think).
Variable: We rerank the top-k documents, where k \in \{10, 20, 50, 100\}.
#### Impact of "Thinking" in Reranking

Hypothesis: Enabling dynamic thinking tokens allows the model to resolve harder logical dependencies in document-query pairs.
Setup:
Retrieval Basis: All four QE settings from Phase 1.
Reranking Depth: Fixed at k=100.
Comparison: We measure the performance delta between Gemini-2.5-Flash (Think) and Gemini-2.5-Flash (No-Think).
#### Impact of Model Strength in Reranking

Hypothesis: Stronger base models (Pro) outperform optimized smaller models (Flash), even when the smaller models utilize thinking.
Setup:
Retrieval Basis: Fixed to the high-performing baseline of BM25 + QE (Gemini-2.5-Flash-Think).
Reranking Depth: Fixed at k=100.
Variable: We compare the four ranking models: Flash-Lite, Flash-No-Think, Flash-Think, and Pro.
[Placeholder: We will insert a diagram here illustrating the matrix of experiments: QE variations on the X-axis vs. RR variations on the Y-axis.]

<div class="l-page">
  <iframe id="ndcg-iframe" src="{{ '/assets/html/2026-04-27-trade-offs-in-llm-compute-for-reasoning-intensive-information-retrieval/cost_vs_ndcg.html' | relative_url }}" frameborder='0' scrolling='no' width="100%" style="border: 0; overflow: hidden; display: block; height: 1px;"></iframe>
</div>

<script>
(function() {
  function resizeIframe() {
    var iframe = document.getElementById('ndcg-iframe');
    if (!iframe) return;
    try {
      var doc = iframe.contentDocument || (iframe.contentWindow && iframe.contentWindow.document);
      if (!doc) return;
      var newHeight = Math.max(
        doc.body ? doc.body.scrollHeight : 0,
        doc.documentElement ? doc.documentElement.scrollHeight : 0
      );
      if (newHeight && newHeight !== parseInt(iframe.style.height, 10)) {
        iframe.style.height = newHeight + 'px';
      }
    } catch (e) {
      // cross-origin guard (should not happen for same-site embeds)
    }
  }
  // Resize on key lifecycle events
  window.addEventListener('load', resizeIframe);
  window.addEventListener('resize', function() { setTimeout(resizeIframe, 100); });
  document.addEventListener('DOMContentLoaded', resizeIframe);
  var el = document.getElementById('ndcg-iframe');
  if (el) el.addEventListener('load', resizeIframe);
})();
</script>



# ... your blog post's content ...