"""
    1. ted_id - TED domain identifier in the format AF-<UniProtID>-F1-model_v4_TED<domain_number_in_chain> i.e. AF-A0A1V6M2Y0-F1-model_v4_TED03
    2. md5_domain - md5 hash of domain sequence
    3. consensus_level - medium (2 methods agreement) or high (3 methods agreement)
    4. chopping - domain boundaries in the format <start>-<stop> or <start>-<stop>_<start>-<stop> for discontinuous domains
    5. nres_domain - number of residues in domain
    6. num_segments - number of individual segments in domain. 
    7. plddt - average pLDDT for domain (range from 0 to 100)
    8. num_helix_strand_turn - number of helix strand turns predicted by STRIDE
    9. num_helix - number of helices predicted by STRIDE
    10. num_strand - number of strands predicted by STRIDE
    11. num_helix_strand - number of helices and strands predicted by STRIDE
    12. num_turn - number of turns predicted by STRIDE
    13. proteome_id - proteome identifier in the format proteome-tax_id-<taxonID>-<shard>_v4 i.e. proteome-tax_id-67581-0_v4
    14. cath_label - CATH superfamily code if predicted, either a C.A.T.H. homologous superfamily or C.A.T. fold assignment. i.e. 3.40.50.300. Otherwise '-'
    15. cath_assignment_level - H for homologous superfamily assignment, T for fold level assignment. Otherwise '-'
    16. cath_assignment_method - Method used to assign a CATH label, either foldseek or foldclass. Otherwise '-'
    17. packing_density - metric used to determine globularity. A domain with packing_density >=10.333 and norm_rg below 0.356 is considered globular
    18. norm_rg - normalised radius of gyration. A domain with packing_density >=10.333 AND norm_rg below 0.356 is considered globular. 
    19. tax_common_name - Common name for organism
    20. tax_scientific_name - Scientific name for organism
    21. tax_lineage - Full taxonomic lineage.
"""

with open("ted_365m.domain_summary.cath.globularity.taxid.tsv", "r") as f:
    lines = [next(f) for _ in range(1000000)]

print("Length of extracted lines:", len(lines))
# print(lines[0])
for s in lines[0].split("\t"):
    print(s)
# print("First 5 rows:")
# domain = ""
# max_len_domain = 0
# max_num_domains = 0
# max_domain_name = ""
# for line in lines:
#     domain_name = line.split("\t")[0]
#     domain = line.split("\t")[1]
#     len_domain, num_domains = calculate_length(domain)
#     if len_domain > max_len_domain:
#         max_len_domain = len_domain
#         max_domain_name = domain_name
#         max_domain = domain
#     if num_domains > max_num_domains:
#         max_num_domains = num_domains
#         max_domain_name = domain_name
#         max_domain = domain
#     # print(domain, len_domain, num_domains)
# print("Max domain:", max_domain_name)
# print("Max domain:", max_domain)
# print("Max length of domain:", max_len_domain)
# print("Max number of domains:", max_num_domains)

# print(df['sequence'].head())
# print(df['sequence'].shape)
# print(df['sequence'].columns)

# print(df['sequence'].head())
# print(df['sequence'].shape)
# print(df['sequence'].columns)