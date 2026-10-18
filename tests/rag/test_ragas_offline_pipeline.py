"""Exercise installed RAGAS evaluation without remote inference or downloads."""
import json

import pytest

from evalscope.backend.rag_eval.ragas.task_template import rag_eval


def test_real_ragas_faithfulness_pipeline_offline(tmp_path, monkeypatch):
    from ragas.llms.base import InstructorLLM

    def generate(self, prompt, response_model):
        if response_model.__name__ == 'StatementGeneratorOutput':
            return response_model(statements=['Paris is in France.'])
        if response_model.__name__ == 'NLIStatementOutput':
            return response_model(statements=[{
                'statement': 'Paris is in France.', 'reason': 'Supported by context.', 'verdict': 1}])
        raise AssertionError(f'Unexpected response schema: {response_model.__name__}')

    monkeypatch.setattr(InstructorLLM, 'generate', generate)
    monkeypatch.setenv('RAGAS_DO_NOT_TRACK', 'true')
    path = tmp_path / 'input.json'
    path.write_text(json.dumps([{
        'user_input': 'Where is Paris?', 'response': 'Paris is in France.',
        'retrieved_contexts': ['Paris is in France.']}]))
    result = rag_eval({
        'testset_file': str(path), 'metrics': ['faithfulness'], 'raise_exceptions': True,
        'critic_llm': {'model_name': 'offline', 'api_key': 'offline-placeholder',
                       'api_base': 'http://127.0.0.1:1/v1'}})
    assert result == {'faithfulness': pytest.approx(1.0)}
    summary = json.loads((tmp_path / 'input_score_summary.json').read_text())
    assert summary['counts']['faithfulness'] == {'valid': 1, 'failed': 0, 'total': 1}
