import argparse
import json
import sys
from pathlib import Path
import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from domain_query.config_utils import load_config
from domain_query.decode import decode_batch
from domain_query.model import DomainQueryModel
from residue_mlm.labels import AA_TO_ID, UNK_ID
from residue_mlm.predict_single import vocabs_from_checkpoint


def load_model_and_vocabs(checkpoint_path: str, config_path: str):
    args = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cath_vocabs = vocabs_from_checkpoint(checkpoint["cath_vocabs"])

    model = DomainQueryModel(
        cath_vocab_sizes=checkpoint["cath_vocab_sizes"],
        esm_model_name=args.esm_model_name,
        freeze_esm=args.freeze_esm,
        max_len=args.max_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_queries=args.num_queries,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cath_vocabs, args

@torch.no_grad()
def predict_one_sequence(model, cath_vocabs, args, sequence: str) -> dict:
    sequence = sequence.strip().upper()
    if len(sequence) > args.max_len:
        print(f"[WARN] sequence has {len(sequence)} residues, only the first {args.max_len} are used")
        sequence = sequence[: args.max_len]

    device = next(model.parameters()).device
    aa_ids = torch.tensor([[AA_TO_ID.get(aa, UNK_ID) for aa in sequence]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(aa_ids, dtype=torch.bool)
    outputs = model([sequence], aa_ids, attention_mask)

    chopping_stars, confidences = decode_batch(
        outputs,
        torch.tensor([len(sequence)]),
        [0],
        cath_vocabs,
        score_threshold=args.score_threshold,
        max_gap=args.max_gap,
        min_segment_len=args.min_segment_len,
    )
    return {"chopping_star": chopping_stars[0], "confidences": confidences[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--config", default="configs/domain_query_config.yaml")
    parser.add_argument("--checkpoint", default="artifacts/domain_query_checkpoint.pt")
    cli = parser.parse_args()

    model, cath_vocabs, args = load_model_and_vocabs(cli.checkpoint, cli.config)
    if torch.cuda.is_available():
        model = model.cuda()
    print(json.dumps(predict_one_sequence(model, cath_vocabs, args, cli.sequence), indent=2))


if __name__ == "__main__":
    main()
