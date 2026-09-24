
# Release audit

Date: 24 September 2026

The clean package was constructed by copying the current audited mechanism
sources; no algorithm file was rewritten during packaging. Verification on the
release contents produced:

- Python compilation: pass;
- `cnm_swarm_sim.validate_release`: 6/6 checks;
- focused release mechanism tests: 5/5 pass;
- complete working-repository test suite: 42/42 pass.

The focused tests cover measured-interface compilation, composition,
targeted deletion/exact restoration, idempotent transfer with tamper rejection,
source-linked failure evidence and the explicit unseen A+C+B protocol.

The package deliberately excludes figures, rosbag utilities, obsolete runners,
draft manuscripts, cached output and trained checkpoints. Those belong in the
separate data release or a versioned large-file archive.

The main unresolved manuscript/code discrepancy is the sensing/training backend:
the tested reference code is a range-ray/PyBullet implementation, whereas the
hardware-oriented manuscript describes stereo convolutional perception and a
larger Flightmare training regime. The algorithmic CNM state, record, graph,
reliability, circulation and causal-intervention mechanisms are represented;
the missing hardware backend must be released separately before exact hardware
reproduction can be claimed.
