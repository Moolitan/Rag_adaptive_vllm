#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real question + context generator for vLLM baseline testing.

Generates short questions (~50 tokens) with variable-length context documents
to simulate RAG workloads and avoid KV cache pollution.
"""

import random
from typing import List, Tuple

# Pool of diverse topics
TOPICS = [
    "artificial intelligence", "climate change", "quantum computing", "space exploration",
    "renewable energy", "genetic engineering", "blockchain technology", "neuroscience",
    "ocean conservation", "urban planning", "cyber security", "machine learning",
    "sustainable agriculture", "autonomous vehicles", "virtual reality", "biotechnology",
    "archaeological discoveries", "economic policy", "educational reform", "public health",
    "wildlife conservation", "social media", "philosophical ethics", "historical events",
    "architectural design", "musical composition", "literary analysis", "political systems",
    "medical research", "environmental law", "data privacy", "cultural anthropology",
    "mathematical theory", "psychological studies", "religious philosophy", "art history",
    "technological innovation", "scientific methodology", "business strategy", "legal frameworks",
    "sports analytics", "culinary arts", "fashion design", "film production",
    "language evolution", "demographic trends", "financial markets", "industrial engineering",
    "agricultural technology", "marine biology", "astronomy research", "geological surveys"
]

# Short question templates (will generate ~30-70 tokens)
SHORT_QUESTIONS = [
    "What are the main principles of {}?",
    "How does {} impact modern society?",
    "What are the key challenges in {}?",
    "Can you explain the fundamentals of {}?",
    "What are recent developments in {}?",
    "How is {} applied in practice?",
    "What are the benefits and risks of {}?",
    "Who are the leading researchers in {}?",
    "What is the current state of {}?",
    "How does {} relate to sustainability?",
]

# Document content building blocks
DOCUMENT_INTROS = [
    "Recent studies have shown that",
    "According to leading researchers in the field",
    "Historical evidence suggests that",
    "Contemporary analysis indicates that",
    "Empirical data demonstrates that",
    "Scientific investigations reveal that",
    "Expert consensus suggests that",
    "Longitudinal studies have found that",
    "Cross-cultural research shows that",
    "Meta-analyses indicate that",
]

DOCUMENT_BODIES = [
    "the fundamental principles underlying this domain are rooted in a complex interplay of theoretical frameworks and practical applications. The evolution of methodologies has been shaped by technological advancements, societal needs, and interdisciplinary collaboration. Key stakeholders including researchers, practitioners, and policy makers have contributed to developing robust approaches that address both immediate challenges and long-term sustainability goals.",
    "significant progress has been made in understanding the underlying mechanisms and causal relationships that govern this field. Breakthrough discoveries have opened new avenues for exploration, while also highlighting critical knowledge gaps that require further investigation. The integration of diverse perspectives from multiple disciplines has enriched our comprehension and enabled more nuanced approaches to problem-solving.",
    "the landscape has been fundamentally transformed by innovations in technology, methodology, and conceptual frameworks. These developments have catalyzed paradigm shifts in how we approach research questions, design interventions, and evaluate outcomes. The convergence of theoretical insights and empirical evidence has strengthened the foundation for evidence-based practice and informed decision-making across various contexts.",
    "emerging patterns reveal complex dynamics that challenge conventional assumptions and demand sophisticated analytical approaches. The interplay between individual factors and systemic influences creates multifaceted scenarios that require careful consideration of contextual variables. Advanced methodologies combining quantitative rigor with qualitative depth have enhanced our ability to capture nuanced phenomena and generate actionable insights.",
    "the field continues to evolve in response to shifting priorities, technological capabilities, and societal demands. Innovation cycles have accelerated, bringing forth novel solutions while also raising important questions about ethical implications, equity considerations, and long-term sustainability. Stakeholder engagement and participatory approaches have become increasingly central to ensuring that developments align with diverse needs and values.",
]

DOCUMENT_DETAILS = [
    "Specifically, researchers have identified several critical factors that influence outcomes, including environmental conditions, resource availability, stakeholder engagement, and institutional support structures. Case studies from diverse geographical regions demonstrate both universal patterns and context-specific variations that inform adaptive strategies.",
    "Key findings include the identification of threshold effects, non-linear relationships, and feedback loops that shape system behavior. These insights have important implications for intervention design, risk assessment, and policy formulation. Comparative analyses across different settings reveal both convergent trends and divergent pathways that reflect local conditions and priorities.",
    "Implementation challenges often arise from misalignment between theoretical models and practical constraints, highlighting the importance of iterative approaches that allow for continuous learning and adaptation. Success factors identified across multiple contexts include strong leadership, adequate resources, stakeholder buy-in, and flexible frameworks that can accommodate local variations.",
    "Measurement and evaluation frameworks have been refined to capture both intended and unintended consequences, proximal and distal outcomes, and short-term and long-term impacts. Multi-method approaches combining quantitative metrics with qualitative insights provide comprehensive understanding of complex phenomena and support evidence-informed decision-making.",
    "Future directions involve scaling proven approaches, addressing persistent inequities, leveraging technological innovations, and strengthening collaborative networks. Emerging priorities include climate resilience, social justice, inclusive participation, and sustainable development that balances economic, social, and environmental objectives.",
]

DOCUMENT_CONCLUSIONS = [
    "In conclusion, the evidence base continues to grow, providing increasingly robust foundations for practice and policy. However, significant challenges remain, including addressing knowledge gaps, bridging theory and practice, ensuring equitable access, and adapting to rapidly changing conditions.",
    "Looking forward, the field faces both opportunities and challenges as it seeks to expand impact while maintaining rigor and relevance. Key priorities include fostering innovation, building capacity, strengthening partnerships, and ensuring that developments serve the broader public good.",
    "Overall, progress has been substantial, yet much work remains to fully realize the potential of current knowledge and capabilities. Success will require sustained commitment, adequate investment, collaborative approaches, and willingness to learn from both successes and setbacks.",
    "The trajectory of development suggests continued advancement, though the pace and direction will depend on multiple factors including resource allocation, policy priorities, technological capabilities, and societal values. Adaptive management and ongoing evaluation will be essential.",
    "Ultimately, the goal is to translate knowledge into meaningful impact that improves outcomes, enhances well-being, and contributes to sustainable development. Achieving this vision requires integration of evidence, engagement of stakeholders, and commitment to continuous improvement.",
]


def generate_short_question(request_id: int, seed: int = None) -> str:
    """
    Generate a short question (~30-70 tokens).

    Args:
        request_id: Unique request identifier
        seed: Random seed for reproducibility

    Returns:
        A short question string
    """
    if seed is not None:
        random.seed(seed + request_id)

    topic = random.choice(TOPICS)
    question_template = random.choice(SHORT_QUESTIONS)
    question = question_template.format(topic)

    # Add request ID
    question_with_id = f"[Question ID: {request_id:05d}] {question}"

    return question_with_id


def generate_context_documents(target_tokens: int, request_id: int, seed: int = None) -> str:
    """
    Generate diverse context documents to reach target token length.

    Args:
        target_tokens: Target length in tokens (~4 chars/token)
        request_id: Unique request identifier
        seed: Random seed for reproducibility

    Returns:
        Context documents string
    """
    if seed is not None:
        random.seed(seed + request_id + 1000)  # Different seed from question

    documents = []
    current_tokens = 0
    target_chars = target_tokens * 4
    doc_id = 1

    # Generate documents until we reach target length
    while current_tokens < target_tokens * 0.90:
        # Pick random components
        topic = random.choice(TOPICS)
        intro = random.choice(DOCUMENT_INTROS)
        body = random.choice(DOCUMENT_BODIES)
        detail = random.choice(DOCUMENT_DETAILS)
        conclusion = random.choice(DOCUMENT_CONCLUSIONS)

        # Build document
        doc = f"Document {doc_id} (Topic: {topic.title()}):\n"
        doc += f"{intro} {body} {detail} {conclusion}"

        # Add more content for very long targets
        if target_tokens > 4000:
            extra_detail = random.choice(DOCUMENT_DETAILS)
            extra_conclusion = random.choice(DOCUMENT_CONCLUSIONS)
            doc += f" {extra_detail} {extra_conclusion}"

        if target_tokens > 8000:
            more_body = random.choice(DOCUMENT_BODIES)
            doc += f" {more_body}"

        documents.append(doc)
        current_tokens = sum(len(d) for d in documents) // 4
        doc_id += 1

        # Safety limit
        if doc_id > 200:
            break

    context = "\n\n".join(documents)
    return context


def generate_rag_prompt(
    context_length: int,
    request_id: int,
    seed: int = None
) -> Tuple[str, str, str]:
    """
    Generate a complete RAG-style prompt: question + context documents.

    Args:
        context_length: Target context length in tokens (not including question)
        request_id: Unique request identifier
        seed: Random seed for reproducibility

    Returns:
        Tuple of (full_prompt, question, context)
    """
    if seed is not None:
        base_seed = seed
    else:
        base_seed = context_length * 1000

    # Generate short question
    question = generate_short_question(request_id, seed=base_seed)

    # Generate context documents
    context = generate_context_documents(context_length, request_id, seed=base_seed)

    # Combine into RAG format
    full_prompt = f"""Based on the following retrieved documents, please answer the question.

Retrieved Documents:
{context}

Question: {question}

Please provide a comprehensive answer based on the documents above."""

    return full_prompt, question, context


# Backward compatibility function
def get_question_for_context_and_request(context_length: int, request_id: int) -> str:
    """
    Get a complete RAG prompt for given context length and request ID.

    This maintains backward compatibility with vllm_baseline.py.

    Args:
        context_length: Target context length in tokens
        request_id: Unique request identifier

    Returns:
        Complete RAG-style prompt string
    """
    seed = context_length * 1000 + request_id
    full_prompt, _, _ = generate_rag_prompt(context_length, request_id, seed=seed)
    return full_prompt


if __name__ == "__main__":
    # Test the generator
    print("=" * 80)
    print("RAG-Style Prompt Generator Test")
    print("=" * 80)

    for ctx in [500, 1000, 2000, 4000, 8000, 16000]:
        print(f"\n{'='*80}")
        print(f"Context length: {ctx} tokens")
        print(f"{'='*80}")

        for req_id in range(3):
            full_prompt, question, context = generate_rag_prompt(ctx, req_id)

            question_tokens = len(question) // 4
            context_tokens = len(context) // 4
            total_tokens = len(full_prompt) // 4

            print(f"\nRequest {req_id}:")
            print(f"  Question: {question[:80]}...")
            print(f"  Question tokens: ~{question_tokens}")
            print(f"  Context tokens:  ~{context_tokens} (target: {ctx})")
            print(f"  Total tokens:    ~{total_tokens}")
            print(f"  Context sample:  {context[:150]}...")
