# In-Context Watermarks for Large Language Models

Official implementation of the LLMs In-Context watermarking techniques presented in the paper:

["In-Context Watermarks for Large Language Models"](https://arxiv.org/abs/2505.16934) by Yepeng Liu, Xuandong Zhao, Christopher Kruegel, Dawn Song and Yuheng Bu.

## Introduction
The growing use of large language models (LLMs) for sensitive applications has highlighted the need for effective watermarking techniques to ensure the provenance and accountability of AI-generated text. However, most existing watermarking methods require access to the decoding process, limiting their applicability in real-world settings. One illustrative example is the use of LLMs by dishonest reviewers in the context of academic peer review, where conference organizers have no access to the model used but still need to detect AI-generated reviews. Motivated by this gap, we introduce In-Context Watermarking (ICW), which embeds watermarks into generated text solely through prompt engineering, leveraging LLMs' in-context learning and instruction-following abilities. We investigate four ICW strategies at different levels of granularity, each paired with a tailored detection method. We further examine the Indirect Prompt Injection (IPI) setting as a specific case study, in which watermarking is covertly triggered by modifying input documents such as academic manuscripts. Our experiments validate the feasibility of ICW as a model-agnostic, practical watermarking approach. Moreover, our findings suggest that as LLMs become more capable, ICW offers a promising direction for scalable and accessible content attribution.


![dts](https://github.com/user-attachments/assets/751eccf2-5b28-4569-bad1-30b2fbce82c6)
*Figure 1. ICWs in Direct Text Stamp Setting. The application of ICW does not require access to the LLM’s decoding process; instead, it relies solely on a predefined watermarking instruction as input. This instruction can be provided either by the user or by a third-party application that interacts with the LLM exclusively through its API to obtain generated text. Once the watermarking instruction is set, users can interact with the LLM as usual, submitting queries and receiving responses, while the generated text automatically contains the embedded invisible watermark.*

![ipi](https://github.com/user-attachments/assets/c38399eb-97cc-4e60-ace8-3bb5f7db766f)
*Figure 2. ICWs in Indirect Prompt Injection Setting (Case study in detecting AI-generated reviews for papers submitted to academic conferences). Conference organizers embed a predefined watermarking instruction (invisible to the reviewer, e.g., ‘white text’) into the submitted papers. Reviewers who input the full PDF into an LLM to generate an AI review, typically a prohibited action, can then be identified by detecting the watermark in the submitted review.*

## Installation
```
# download the code
git clone https://github.com/yepengliu/In-Context-Watermarks.git
cd In-Context-Watermarks

# install the required environment
pip install -r requirements.txt
```

