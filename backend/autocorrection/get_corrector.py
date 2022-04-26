from typing import Optional

from autocorrection.src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator
from autocorrection.src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from autocorrection.src.corrector.beam_search.batched_beam_search_corrector import BatchedBeamSearchCorrector

def get_corrector(approach: str,
                  penalties: Optional[str],
                  insertion_penalty: float = 0,
                  deletion_penalty: float = 0,
                  fwd_model_name: Optional[str] = None,
                  bid_model_name: Optional[str] = None):
    fwd_model = UnidirectionalLMEstimator()
    fwd_model.load(fwd_model_name)
    if bid_model_name is None:
        bid_model = None
    else:
        bid_model = BidirectionalLabelingEstimator()
        bid_model.load(bid_model_name)
    p_ins = insertion_penalty
    p_del = deletion_penalty
    corrector = BatchedBeamSearchCorrector(fwd_model, 
                                           insertion_penalty=-p_ins, 
                                           deletion_penalty=-p_del, 
                                           n_beams=5,
                                           verbose=False, 
                                           labeling_model=bid_model, 
                                           add_epsilon=bid_model is not None)
    return corrector

if __name__ == "__main__":
    approach = "CUSTOM"
    p_ins = 0
    p_del = 0
    penalties = ""
    fwd = "lm/unilm"
    bid = None
    
    tokr = get_corrector(approach, penalties, p_ins, p_del,
                            fwd_model_name=fwd, bid_model_name=bid)
    
    print(tokr.correct("hoomnaytoi dihojc"))