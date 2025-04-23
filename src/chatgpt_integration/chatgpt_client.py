"""
Integration with ChatGPT API for question verification.
"""

import os
import json
import time
import openai
from dotenv import load_dotenv


class ChatGPTIntegration:
    """
    Integration with ChatGPT API for verifying generated questions.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the ChatGPT integration.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will attempt to load from environment.
        """
        # Load environment variables
        load_dotenv()
        
        # Set API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            openai.api_key = self.api_key
        else:
            raise ValueError("OpenAI API key not found. Please provide it or set OPENAI_API_KEY environment variable.")
    
    def verify_question(self, question, tokens=None, model="gpt-3.5-turbo"):
        """
        Verify a generated question using ChatGPT.
        
        Args:
            question (str): The question to verify
            tokens (list, optional): The tokens used to generate the question
            model (str): The OpenAI model to use
            
        Returns:
            dict: Verification results including quality score and feedback
        """
        # Prepare system message
        system_message = "You are an assistant that evaluates the quality of questions."
        
        # Prepare user message
        if tokens:
            user_message = f"Evaluate the following question that was generated based on these tokens: {', '.join(tokens)}.\n\nQuestion: {question}\n\nPlease rate the question on a scale of 1-10 for clarity, relevance to the tokens, and overall quality. Provide brief feedback on how the question could be improved."
        else:
            user_message = f"Evaluate the following question: {question}\n\nPlease rate the question on a scale of 1-10 for clarity and overall quality. Provide brief feedback on how the question could be improved."
        
        # Call ChatGPT API
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            # Extract response
            verification_text = response.choices[0].message.content
            
            # Parse response to extract score
            # This is a simple heuristic; in practice, you might want to use a more robust approach
            score = None
            for line in verification_text.split('\n'):
                if "rating" in line.lower() or "score" in line.lower() or "/10" in line:
                    # Try to extract a number from this line
                    for word in line.split():
                        if word.replace('.', '').isdigit():
                            score = float(word)
                            break
            
            # If we couldn't extract a score, default to 5
            if score is None:
                score = 5.0
            
            return {
                "question": question,
                "score": score,
                "feedback": verification_text,
                "tokens": tokens
            }
            
        except Exception as e:
            return {
                "question": question,
                "score": 0,
                "feedback": f"Error during verification: {str(e)}",
                "tokens": tokens
            }
    
    def verify_questions(self, questions, tokens=None, model="gpt-3.5-turbo"):
        """
        Verify multiple generated questions using ChatGPT.
        
        Args:
            questions (list): List of questions to verify
            tokens (list, optional): The tokens used to generate the questions
            model (str): The OpenAI model to use
            
        Returns:
            list: List of verification results
        """
        results = []
        
        for question in questions:
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            
            # Verify question
            result = self.verify_question(question, tokens, model)
            results.append(result)
        
        return results
    
    def answer_question(self, question, model="gpt-3.5-turbo"):
        """
        Get an answer to a question using ChatGPT.
        
        Args:
            question (str): The question to answer
            model (str): The OpenAI model to use
            
        Returns:
            str: The answer from ChatGPT
        """
        # Prepare system message
        system_message = "You are a helpful assistant that provides informative and accurate answers."
        
        # Call ChatGPT API
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # Extract response
            answer = response.choices[0].message.content
            
            return answer
            
        except Exception as e:
            return f"Error getting answer: {str(e)}"
    
    def evaluate_question_answer_pair(self, question, answer, tokens=None, model="gpt-3.5-turbo"):
        """
        Evaluate a question-answer pair for coherence and quality.
        
        Args:
            question (str): The question
            answer (str): The answer to evaluate
            tokens (list, optional): The tokens used to generate the question
            model (str): The OpenAI model to use
            
        Returns:
            dict: Evaluation results
        """
        # Prepare system message
        system_message = "You are an assistant that evaluates the quality of question-answer pairs."
        
        # Prepare user message
        if tokens:
            user_message = f"Evaluate the following question-answer pair. The question was generated based on these tokens: {', '.join(tokens)}.\n\nQuestion: {question}\n\nAnswer: {answer}\n\nPlease rate the coherence and quality of this question-answer pair on a scale of 1-10. Provide brief feedback on the strengths and weaknesses of both the question and the answer."
        else:
            user_message = f"Evaluate the following question-answer pair:\n\nQuestion: {question}\n\nAnswer: {answer}\n\nPlease rate the coherence and quality of this question-answer pair on a scale of 1-10. Provide brief feedback on the strengths and weaknesses of both the question and the answer."
        
        # Call ChatGPT API
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            # Extract response
            evaluation_text = response.choices[0].message.content
            
            # Parse response to extract score (simple heuristic)
            score = None
            for line in evaluation_text.split('\n'):
                if "rating" in line.lower() or "score" in line.lower() or "/10" in line:
                    for word in line.split():
                        if word.replace('.', '').isdigit():
                            score = float(word)
                            break
            
            # If we couldn't extract a score, default to 5
            if score is None:
                score = 5.0
            
            return {
                "question": question,
                "answer": answer,
                "score": score,
                "feedback": evaluation_text,
                "tokens": tokens
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": answer,
                "score": 0,
                "feedback": f"Error during evaluation: {str(e)}",
                "tokens": tokens
            }
