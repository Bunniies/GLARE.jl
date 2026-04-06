# GLARE Project Memory

- [GLARE project overview](project_glare.md) — L-CNN-style gauge-equivariant NN for config-by-config prediction of lattice QCD rest-eigen correlators C(t)
- [User background in lattice QCD and ML](user_physics_ml.md) — Alessandro Conigli: lattice QCD physicist building gauge-equivariant nets in Julia; assume full domain fluency
- [Loss function ideas for Phase 3](project_loss_ideas.md) — time-weighted MSE and L=(1-r²) direct r-maximizing loss, to revisit in Phase 3 tuning
- [Data scale and IO optimization](project_data_scale.md) — A654 plaq_scalar 30 MB/config; Float32 storage + in-memory preloading needed before full training; GPU for Phase 2+
