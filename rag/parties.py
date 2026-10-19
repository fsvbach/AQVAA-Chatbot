"""Canonical Swiss party labels shared across data loading and classification.

The candidate data (platform reaction batch) and the classifier's party filters use this fixed
set of labels. Manifesto documents are organized in one subfolder per party, and
each folder name is expected to equal its canonical label here — so a manifesto's
party metadata filters on the same strings as candidate data.
"""

# Must match the party values of the candidates uploaded to the platform and the
# party subfolder names under settings.manifestos_path.
CANONICAL_PARTIES = {"EDU", "EVP", "FDP", "GLP", "Grüne", "Mitte", "SP", "SVP", "Sonstige"}