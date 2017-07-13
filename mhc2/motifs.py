# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from Bio import motifs
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
import numpy as np

def compute_positional_weight_matrix(seqs, length=None, pseudocounts=0.5):
    if not length:
        length = min(len(s) for s in seqs)
    alphabet = IUPAC.protein
    bio_seqs = [
        Seq(p[:length], alphabet=alphabet) for p in seqs
    ]
    m = motifs.create(
        bio_seqs,
        alphabet=alphabet)
    return m.counts.normalize(pseudocounts=pseudocounts)

def apply_pssm_to_peptides(pssm, peptides):
    scores = []
    for peptide in peptides:
        aa_scores = []
        for i, aa in enumerate(peptide):
            aa_scores.append(pssm[aa][i])
        scores.append(np.mean(aa_scores))
    return scores

def pssm_goodness_of_fit(pssm, peptides):
    scores = apply_pssm_to_peptides(pssm, peptides)
    return np.mean(scores)

def compatibility_score_for_binding_cores(peptides):
    pwm = compute_positional_weight_matrix(peptides)
    pssm = pwm.log_odds()
    return pssm_goodness_of_fit(pssm, peptides)
