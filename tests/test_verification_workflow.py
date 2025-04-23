"""
Test file for the ChatGPT integration and verification workflow.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
from src.chatgpt_integration.chatgpt_client import ChatGPTIntegration
from src.chatgpt_integration.verification_workflow import VerificationWorkflow, BatchVerificationWorkflow


class TestChatGPTIntegration(unittest.TestCase):
    """Test cases for the ChatGPTIntegration class."""
    
    @patch('src.chatgpt_integration.chatgpt_client.openai')
    @patch('src.chatgpt_integration.chatgpt_client.load_dotenv')
    def setUp(self, mock_load_dotenv, mock_openai):
        """Set up test fixtures with mocked dependencies."""
        # Mock environment variables
        os.environ['OPENAI_API_KEY'] = 'test_api_key'
        
        # Create a ChatGPT integration with mocked dependencies
        self.integration = ChatGPTIntegration()
        
        # Store the mock openai module for later assertions
        self.mock_openai = mock_openai
    
    def test_initialization(self):
        """Test that the integration initializes correctly."""
        self.assertIsInstance(self.integration, ChatGPTIntegration)
        self.assertEqual(self.integration.api_key, 'test_api_key')
    
    def test_verify_question(self):
        """Test the verify_question method."""
        # Mock question and tokens
        question = "What is the relationship between artificial intelligence and ethics?"
        tokens = ["artificial intelligence", "ethics"]
        
        # Mock ChatCompletion.create response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Rating: 8/10. This is a clear and relevant question that addresses the ethical implications of AI."
        self.mock_openai.ChatCompletion.create.return_value = mock_response
        
        # Call the method
        result = self.integration.verify_question(question, tokens)
        
        # Check that openai.ChatCompletion.create was called with the correct arguments
        self.mock_openai.ChatCompletion.create.assert_called_once()
        call_args = self.mock_openai.ChatCompletion.create.call_args[1]
        self.assertEqual(call_args['model'], 'gpt-3.5-turbo')
        self.assertEqual(len(call_args['messages']), 2)
        
        # Check that the result has the expected structure
        self.assertIn('question', result)
        self.assertIn('score', result)
        self.assertIn('feedback', result)
        self.assertIn('tokens', result)
        
        # Check that the result contains the expected values
        self.assertEqual(result['question'], question)
        self.assertEqual(result['score'], 8.0)
        self.assertEqual(result['feedback'], mock_response.choices[0].message.content)
        self.assertEqual(result['tokens'], tokens)
    
    def test_verify_questions(self):
        """Test the verify_questions method."""
        # Mock questions and tokens
        questions = [
            "What is artificial intelligence?",
            "How does ethics relate to AI?"
        ]
        tokens = ["artificial intelligence", "ethics"]
        
        # Mock verify_question method to return predetermined results
        mock_results = [
            {
                'question': questions[0],
                'score': 7.0,
                'feedback': "Good question about AI.",
                'tokens': tokens
            },
            {
                'question': questions[1],
                'score': 8.0,
                'feedback': "Good question about ethics and AI.",
                'tokens': tokens
            }
        ]
        
        with patch.object(self.integration, 'verify_question', side_effect=mock_results):
            # Call the method
            results = self.integration.verify_questions(questions, tokens)
            
            # Check that verify_question was called twice with the correct arguments
            self.assertEqual(self.integration.verify_question.call_count, 2)
            self.integration.verify_question.assert_any_call(questions[0], tokens, 'gpt-3.5-turbo')
            self.integration.verify_question.assert_any_call(questions[1], tokens, 'gpt-3.5-turbo')
            
            # Check that the results match the mock results
            self.assertEqual(results, mock_results)


class TestVerificationWorkflow(unittest.TestCase):
    """Test cases for the VerificationWorkflow class."""
    
    @patch('src.chatgpt_integration.verification_workflow.ChatGPTIntegration')
    @patch('src.chatgpt_integration.verification_workflow.RandomQuestionGenerator')
    @patch('src.chatgpt_integration.verification_workflow.TokenProcessor')
    def setUp(self, mock_token_processor, mock_random_generator, mock_chatgpt_integration):
        """Set up test fixtures with mocked dependencies."""
        # Mock the token processor
        self.mock_token_processor_instance = MagicMock()
        mock_token_processor.return_value = self.mock_token_processor_instance
        
        # Mock the random generator
        self.mock_random_generator_instance = MagicMock()
        mock_random_generator.return_value = self.mock_random_generator_instance
        
        # Mock the ChatGPT integration
        self.mock_chatgpt_integration_instance = MagicMock()
        mock_chatgpt_integration.return_value = self.mock_chatgpt_integration_instance
        
        # Create a verification workflow with mocked dependencies
        self.workflow = VerificationWorkflow(use_random_generator=True)
    
    def test_initialization(self):
        """Test that the workflow initializes correctly."""
        self.assertIsInstance(self.workflow, VerificationWorkflow)
    
    @patch('src.chatgpt_integration.verification_workflow.os.makedirs')
    @patch('src.chatgpt_integration.verification_workflow.open')
    @patch('src.chatgpt_integration.verification_workflow.json.dump')
    def test_generate_and_verify(self, mock_json_dump, mock_open, mock_makedirs):
        """Test the generate_and_verify method."""
        # Mock tokens and questions
        tokens = ["artificial intelligence", "ethics"]
        questions = [
            "What is the relationship between artificial intelligence and ethics?",
            "How does ethics influence AI development?"
        ]
        
        # Mock the random generator's generate_questions method
        self.mock_random_generator_instance.generate_questions.return_value = questions
        
        # Mock the ChatGPT integration's verify_questions method
        mock_verifications = [
            {
                'question': questions[0],
                'score': 7.0,
                'feedback': "Good question about AI and ethics.",
                'tokens': tokens
            },
            {
                'question': questions[1],
                'score': 8.0,
                'feedback': "Good question about ethics influencing AI.",
                'tokens': tokens
            }
        ]
        self.mock_chatgpt_integration_instance.verify_questions.return_value = mock_verifications
        
        # Mock the ChatGPT integration's answer_question method
        mock_answers = ["Answer 1", "Answer 2"]
        self.mock_chatgpt_integration_instance.answer_question.side_effect = mock_answers
        
        # Mock the ChatGPT integration's evaluate_question_answer_pair method
        mock_evaluations = [
            {
                'question': questions[0],
                'answer': mock_answers[0],
                'score': 7.5,
                'feedback': "Good question-answer pair.",
                'tokens': tokens
            },
            {
                'question': questions[1],
                'answer': mock_answers[1],
                'score': 8.5,
                'feedback': "Excellent question-answer pair.",
                'tokens': tokens
            }
        ]
        self.mock_chatgpt_integration_instance.evaluate_question_answer_pair.side_effect = mock_evaluations
        
        # Call the method
        results = self.workflow.generate_and_verify(tokens, num_questions=2)
        
        # Check that the random generator's generate_questions method was called with the tokens
        self.mock_random_generator_instance.generate_questions.assert_called_once_with(tokens, 2)
        
        # Check that the ChatGPT integration's verify_questions method was called with the questions and tokens
        self.mock_chatgpt_integration_instance.verify_questions.assert_called_once_with(questions, tokens)
        
        # Check that the ChatGPT integration's answer_question method was called for each question
        self.assertEqual(self.mock_chatgpt_integration_instance.answer_question.call_count, 2)
        self.mock_chatgpt_integration_instance.answer_question.assert_any_call(questions[0])
        self.mock_chatgpt_integration_instance.answer_question.assert_any_call(questions[1])
        
        # Check that the ChatGPT integration's evaluate_question_answer_pair method was called for each pair
        self.assertEqual(self.mock_chatgpt_integration_instance.evaluate_question_answer_pair.call_count, 2)
        
        # Check that the result has the expected structure
        self.assertIn('tokens', results)
        self.assertIn('questions', results)
        self.assertIn('verifications', results)
        self.assertIn('question_answer_pairs', results)
        
        # Check that the result contains the expected values
        self.assertEqual(results['tokens'], tokens)
        self.assertEqual(results['questions'], questions)
        self.assertEqual(results['verifications'], mock_verifications)
        self.assertEqual(results['question_answer_pairs'], mock_evaluations)
        
        # Check that the results were saved
        mock_makedirs.assert_called_once()
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()


class TestBatchVerificationWorkflow(unittest.TestCase):
    """Test cases for the BatchVerificationWorkflow class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the verification workflow
        self.mock_verification_workflow = MagicMock()
        
        # Create a batch verification workflow with the mocked verification workflow
        self.batch_workflow = BatchVerificationWorkflow(self.mock_verification_workflow)
    
    def test_initialization(self):
        """Test that the batch workflow initializes correctly."""
        self.assertIsInstance(self.batch_workflow, BatchVerificationWorkflow)
    
    @patch('src.chatgpt_integration.verification_workflow.os.makedirs')
    @patch('src.chatgpt_integration.verification_workflow.open')
    @patch('src.chatgpt_integration.verification_workflow.json.dump')
    def test_process_token_sets(self, mock_json_dump, mock_open, mock_makedirs):
        """Test the process_token_sets method."""
        # Mock token sets
        token_sets = [
            ["artificial intelligence", "ethics"],
            ["quantum computing", "algorithms"],
            ["blockchain", "cryptocurrency"]
        ]
        
        # Mock the verification workflow's generate_and_verify method
        mock_results = [
            {
                'tokens': token_sets[0],
                'questions': ["Question 1", "Question 2"],
                'verifications': [{'score': 7.0}, {'score': 8.0}]
            },
            {
                'tokens': token_sets[1],
                'questions': ["Question 3", "Question 4"],
                'verifications': [{'score': 6.0}, {'score': 9.0}]
            },
            {
                'tokens': token_sets[2],
                'questions': ["Question 5", "Question 6"],
                'verifications': [{'score': 7.5}, {'score': 8.5}]
            }
        ]
        self.mock_verification_workflow.generate_and_verify.side_effect = mock_results
        
        # Call the method
        results = self.batch_workflow.process_token_sets(token_sets, num_questions_per_set=2)
        
        # Check that the verification workflow's generate_and_verify method was called for each token set
        self.assertEqual(self.mock_verification_workflow.generate_and_verify.call_count, 3)
        for i, token_set in enumerate(token_sets):
            self.mock_verification_workflow.generate_and_verify.assert_any_call(
                token_set, num_questions=2, save_results=False
            )
        
        # Check that the results match the mock results
        self.assertEqual(results, mock_results)
        
        # Check that the batch results were saved
        mock_makedirs.assert_called_once()
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()


if __name__ == '__main__':
    unittest.main()
