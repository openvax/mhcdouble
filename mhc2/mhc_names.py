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

from collections import defaultdict

# source: wikipedia article on HLA-DQ
dq_allele_freq_table = """
05:01   02:01   13.16
02:01   02:02   11.08
03:02   02:02   0.08
03:01   04:02   0.03
03:02   04:02   0.11
04:01   04:02   2.26
01:01   05:01   10.85
01:02   05:01   0.03
01:03   05:01   0.03
01:04   05:01   0.71
01:02   05:02   1.20
01:03   05:02   0.05
01:04   05:03   2.03
01:02   05:04   0.08
01:03   06:01   0.66
01:02   06:02   14.27
01:03   06:02   0.03
01:04   06:02   0.03
01:02   06:03   0.27
01:03   06:03   5.66
01:02   06:04   3.40
01:02   06:09   0.71
02:01   03:01   0.05
03:01   03:01   0.16
03:03   03:01   6.45
03:01   03:04   0.09
03:02   03:04   0.09
04:01   03:01   0.03
05:05   03:01   11.06
06:01   03:01   0.11
03:01   03:02   9.62
03:02   03:02   0.93
02:01   03:03   3.66
03:02   03:03   0.79
"""

# map each beta to a list of (alpha, freq) pairs
dq_beta_to_alphas = defaultdict(list)
dq_alpha_to_betas = defaultdict(list)

for line in dq_allele_freq_table.split("\n"):
    line = line.strip()
    if not line:
        continue
    a, b, freq = line.split()
    a = a.replace(":", "")
    b = b.replace(":", "")
    freq = float(freq)
    dq_beta_to_alphas[b].append((a, freq))
    dq_alpha_to_betas[a].append((b, freq))

dq_beta_to_alpha = {
    b: max(pairs, key=lambda x: x[1])[0]
    for (b, pairs) in dq_beta_to_alphas.items()
}

dq_alpha_to_beta = {
    a: max(pairs, key=lambda x: x[1])[0]
    for (a, pairs) in dq_alpha_to_betas.items()
}

def normalize_mhc_name(name):
    if len(name) == 7 and name.startswith("DRB"):
        name = "DRB1" + name[3:]
    if not name.startswith("HLA"):
        name = "HLA-" + name
    name = name.replace("_", "")
    name = name.replace(":", "")
    name = name.replace("/", "-")
    name = name.replace("*", "")
    if name.startswith("HLA-DRB"):
        # if DR beta is given without its alpha chain
        gene_number = name[7]
        beta = name[8:]
        name = "HLA-DRA10101-DRB%s%s" % (gene_number, beta,)
    elif name.startswith("HLA-DRA"):
        # expand DRA to DRA1
        suffix = name[7:]
        parts = suffix.split("-")
        if len(parts[0]) == 4:
            name = "HLA-DRA1%s" % ("-".join(parts),)
    elif name.startswith("HLA-DQB1"):
        # if DQ beta is given without its paired alpha
        assert len(name) == 12, (len(name), name)
        beta = name[8:12]
        if beta in dq_beta_to_alpha:
            name = "HLA-DQA1%s-DQB1%s" % (
                dq_beta_to_alpha[beta], beta)
    elif name.startswith("HLA-DQA0"):
        # if DQA1 has been written as just DQA
        parts = name.split("-")
        alpha = parts[1]
        name = "-".join(["HLA", "DQA1%s" % alpha[3:]] + parts[2:])

    if name.startswith("HLA-DQA1"):
        # if DQ alpha is given without its paired beta
        parts = name.split("-")
        if len(parts) == 2:
            alpha = parts[1][4:]
            beta = dq_alpha_to_beta[alpha]
            name = "HLA-DQA1%s-DQB1%s" % (alpha, beta)
    return name

test_normalized = normalize_mhc_name("HLA-DRA*0101/DRB1*0801")
assert test_normalized == "HLA-DRA10101-DRB10801", test_normalized

def normalize_mhc_names(alleles):
    """
    Normalizes list of allele names
    """
    # first group alleles to minimize the number of calls to the
    # normalization function
    allele_to_indices = defaultdict(list)
    for i, allele in enumerate(alleles):
        allele_to_indices[allele].append(i)
    results = [None] * len(alleles)
    for allele, indices in allele_to_indices.items():
        allele = normalize_mhc_name(allele)
        for i in indices:
            results[i] = allele
    return results


