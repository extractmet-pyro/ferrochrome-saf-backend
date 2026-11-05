import time


def test_simulation_throughput(sim_factory):
    s = sim_factory()
    t = time.time()
    for _ in range(60):
        s.step()
    dt = time.time() - t
    print(f"\nPERF sim: {60 / dt:.1f} steps/s ({60 * s.dt / 3600 / dt:.0f} simulated h per wall s)")
    assert dt < 30


def test_static_optimiser_walltime():
    from app.domain.material_streams import DEFAULT_RECIPE, default_materials
    from app.domain.static_balance import static_balance
    t = time.time()
    for _ in range(20):
        static_balance(DEFAULT_RECIPE, default_materials())
    dt = (time.time() - t) / 20
    print(f"\nPERF static balance: {1000 * dt:.1f} ms/eval")
    assert dt < 1.0
