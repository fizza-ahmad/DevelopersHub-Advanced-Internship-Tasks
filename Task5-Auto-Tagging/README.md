# Task 5: Auto Tagging Support Tickets Using LLM

## Objective
This project automatically classifies support tickets into relevant categories using a Large Language Model (LLM).

## Methodology
- Used Zero-shot classification with HuggingFace model (facebook/bart-large-mnli)
- Applied Few-shot learning using example prompts
- Compared both approaches for better accuracy understanding

## Labels Used
- Billing Issue  
- Technical Issue  
- Password Reset  
- Account Access  
- Refund Request  
- Delivery Issue  
- Complaint  
- Subscription Issue  

## Approach
1. Load pretrained LLM model
2. Define ticket categories
3. Perform zero-shot classification
4. Improve results using few-shot prompting
5. Compare outputs

## Tools Used
- Python
- HuggingFace Transformers
- PyTorch
- Google Colab

## Result
The model successfully predicts top 3 relevant tags for each support ticket using zero-shot and few-shot learning.