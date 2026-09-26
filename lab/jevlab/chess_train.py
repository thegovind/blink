"""ChessBench-format training rows from searchless_chess TRAIN behavioral-cloning data (Stockfish best move).

Excludes every position whose key (board, side to move, castling, en passant) occurs in the suite's
ChessBench test rows. Row format mirrors the suite's task format (state fen/board/side_to_move; all legal
moves as options keyed by UCI, ordered by a hash; gold = Stockfish move).
  python -m jevlab.chess_train BAG OUT.jsonl N [SUITE_ROWS]
"""
import gzip
import hashlib
import json
import mmap
import random
import struct
import sys


def poskey(fen):
    return " ".join(fen.split()[:4])


def main():
    import chess

    bag, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    suite = sys.argv[4] if len(sys.argv) > 4 else "decision-index/suite/selected-rows.jsonl.gz"
    test = set()
    for line in gzip.open(suite, "rt"):
        r = json.loads(line)
        if r["_evaluation"].get("dataset") == "ChessBench":
            test.add(poskey(r["state"]["fen"]))
    print("test positions", len(test), flush=True)
    rng = random.Random("chess-train-v1")
    kept = seen = excl = 0
    keys = set()
    with open(bag, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf, open(out, "w") as g:
        index = struct.unpack_from("<Q", buf, len(buf) - 8)[0]
        nrec = (len(buf) - 8 - index) // 8
        print("records", nrec, flush=True)
        while kept < n:
            i = rng.randrange(nrec)
            start = struct.unpack_from("<Q", buf, index + 8 * (i - 1))[0] if i > 0 else 0
            end = struct.unpack_from("<Q", buf, index + 8 * i)[0]
            rec = buf[start:end]
            length = shift = off = 0
            while True:
                b = rec[off]; off += 1
                length |= (b & 127) << shift
                if b < 128:
                    break
                shift += 7
            fen = rec[off: off + length].decode()
            move = rec[off + length:].decode()
            seen += 1
            k = poskey(fen)
            if k in test:
                excl += 1
                continue
            if k in keys:
                continue
            board = chess.Board(fen)
            moves = {m.uci(): m for m in board.legal_moves}
            if move not in moves or not 2 <= len(moves) <= 255:
                continue
            keys.add(k)
            order = sorted(moves, key=lambda mv: hashlib.sha256((fen + ":" + mv).encode()).digest())
            crit = {mv: {"uci": mv, "san": board.san(moves[mv])} for mv in order}
            sid = hashlib.sha256(fen.encode()).hexdigest()[:16]
            g.write(json.dumps({"id": f"chess:{sid}", "src": "chess_bc", "state": {"fen": fen, "board": str(board), "side_to_move": "white" if board.turn else "black"},
                                "question": {"type": "choice", "instructions": "Choose the strongest legal move for the side to move. Board rows run from rank 8 to rank 1; columns a to h. Uppercase pieces are white. All legal moves are supplied.",
                                             "criteria": crit}, "gold": move, "target": None,
                                "meta": {"group": f"chess:{sid}", "legal_moves": len(moves)}}) + "\n")
            kept += 1
    print(f"kept {kept} (sampled {seen}, excluded as test positions {excl})")


if __name__ == "__main__":
    main()
