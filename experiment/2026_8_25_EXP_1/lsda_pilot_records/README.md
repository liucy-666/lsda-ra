# LSDA: strict one-hot local specialist diffusion

This folder contains the clean-room implementation requested for the LSDA test.

Pipeline:

1. Generate one native short-short (SS) image.
2. Segment the visible SS instances with SAM. Candidate masks are selected jointly by
   predicted quality, connectedness, position, border contact, area, and mutual overlap.
3. Convert the cleaned image-space masks into an exhaustive one-hot latent partition:
   one owner per entity plus the native-SS scaffold as the background owner.
4. Restart from the same initial seed noise. At every denoising step, all specialists
   read the same current latent. Each entity receives only its standalone Short phrase.
   Entity candidate states are committed only inside their masks; the mask complement is
   copied from the same-seed native-SS trajectory at the matching step.

The implementation records the partition extrema, out-of-owner committed-write energy,
and exact background-to-native-SS matching error at every step. It never uses standalone,
SL, or LL images/latents as donors.

Server workspace: `/science/wx/pry/LSDA`

Local workspace: `D:\Python\MMDIT\LSDA`
