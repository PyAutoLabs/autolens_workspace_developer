def fit():
    """
    SLaM (Source, Light and Mass): Light Parametric + Mass Total + Source Parametric
    ================================================================================

    SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
    lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
    which customize the model and analysis in that pipeline.

    The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
    uses a parametric `Sersic` profile for the bulge, this will be used in the subsequent MASS LIGHT DARK PIPELINE.

    Using a SOURCE LP PIPELINE, LIGHT PIPELINE and a MASS LIGHT DARK PIPELINE this SLaM script fits `Imaging` of
    a strong lens system, where in the final model:

     - The lens galaxy's light is a bulge+disk `Sersic` and `Sersic`.
     - The lens galaxy's stellar mass distribution is a bulge+disk tied to the light model above.
     - The lens galaxy's dark matter mass distribution is modeled as a `NFWMCRLudlow`.
     - The source galaxy's light is a parametric `Inversion`.

    This runner uses the SLaM pipelines:

     `source_lp`
     `source_pix/with_lens_light`
     `light_lp`
     `mass_total/mass_light_dark`

    Check them out for a detailed description of the analysis!
    """

    # %matplotlib inline
    # from pyprojroot import here
    # workspace_path = str(here())
    # %cd $workspace_path
    # print(f"Working Directory has been set to `{workspace_path}`")

    import os
    from os import path
    from pathlib import Path

    import autofit as af
    import autolens as al
    import autolens.plot as aplt
    import slam_pipeline

    """
    __Dataset__ 
    
    Load the `Imaging` data, define the `Mask2D` and plot them.
    """
    dataset_name = "with_lens_light"
    dataset_path = path.join("dataset", "imaging", dataset_name)

    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        pixel_scales=0.2,
    )

    mask_radius = 3.0

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )

    dataset = dataset.apply_mask(mask=mask)

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.1, 0.3],
    )

    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size,
    )

    # dataset = dataset.apply_sparse_operator()

    """
    __Settings AutoFit__
    
    The settings of autofit, which controls the output paths, parallelization, database use, etc.
    """
    settings_search = af.SettingsSearch(
        path_prefix=Path("slam") / "source_pix" / "mass_light_dark" / "base",
        session=None,
    )

    """
    __Redshifts__
    
    The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
    from arc-seconds to kiloparsecs, masses to solar masses, etc.).
    """
    redshift_lens = 0.5
    redshift_source = 1.0

    """
    __SOURCE LP PIPELINE (with lens light)__
    
    The SOURCE LP PIPELINE (with lens light) uses three searches to initialize a robust model for the 
    source galaxy's light, which in this example:
    
     - Uses a parametric `Sersic` bulge and `Exponential` disk with centres aligned for the lens
     galaxy's light.
    
     - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.
    
     __Settings__:
    
     - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS TOTAL PIPELINE).
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        use_jax=True,
    )

    # Lens Light

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=30,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
    )

    mass = af.Model(al.mp.Isothermal)

    # Source:

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    source_lp_result = slam_pipeline.source_lp.run(
        settings_search=settings_search,
        analysis=analysis,
        lens_bulge=lens_bulge,
        lens_disk=None,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear),
        source_bulge=source_bulge,
        mass_centre=(0.0, 0.0),
        redshift_lens=redshift_lens,
        redshift_source=redshift_source,
    )

    """
    __Mesh Shape__
    
    The `mesh_shape` parameter defines number of pixels used by the rectangular mesh to reconstruct the source,
    set below to 28 x 28. 
    
    The `mesh_shape` must be fixed before modeling and cannot be a free parameter of the model, because JAX uses the
    mesh shape to define static shaped arrays which use the mesh to reconstruct the source. For a rectangular
    mesh, the same number of pixels must be used in the y and x directions.
    
    __Edge Zeroing__
    
    By default, all pixels at the edge of the mesh in the source-plane are forced to solutions of zero brightness by 
    the linear algebra solver. This prevents unphysical solutions where pixels at the edge of the mesh reconstruct 
    bright surface brightnesses, often because they fit residuals from the lens light subtraction.
    
    For a rectangular mesh, the source code computes edge pixels internally using the known
    pixels at the edge of the mesh. 
    """
    mesh_pixels_yx = 28
    mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

    """
    __SOURCE PIX PIPELINE__

    The SOURCE PIX PIPELINE is identical to the `slam_start_here.ipynb` example.
    """
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_lp_result.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
    )

    source_pix_result_1 = slam_pipeline.source_pix.run_1(
        settings_search=settings_search,
        analysis=analysis,
        source_lp_result=source_lp_result,
        mesh_init=af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape),
        regularization_init=al.reg.Adapt,
    )

    """
    __SOURCE PIX PIPELINE 2__

    The SOURCE PIX PIPELINE 2 is identical to the `slam_start_here.ipynb` example.
    """
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
    )

    source_pix_result_2 = slam_pipeline.source_pix.run_2(
        settings_search=settings_search,
        analysis=analysis,
        source_lp_result=source_lp_result,
        source_pix_result_1=source_pix_result_1,
        mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
        regularization=al.reg.Adapt,
    )

    """
    __LIGHT LP PIPELINE__
    
    The LIGHT LP PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
    lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE PIX PIPELINE.
    In this example it:
    
     - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens galaxy's 
     light [Do not use the results of the SOURCE LP PIPELINE to initialize priors].
    
     - Uses an `Isothermal` model for the lens's total mass distribution [fixed from SOURCE LP PIPELINE].
    
     - Uses an `Inversion` for the source's light [priors fixed from SOURCE PIX PIPELINE].
    
     - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
     PIPELINE [fixed values].
    """
    bulge = af.Model(al.lp.Sersic)
    disk = af.Model(al.lp.Sersic)
    bulge.centre = disk.centre

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
    )

    light_result = slam_pipeline.light_lp.run(
        settings_search=settings_search,
        analysis=analysis,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        lens_bulge=bulge,
        lens_disk=disk,
    )

    """
    __MASS LIGHT DARK PIPELINE (with lens light)__
    
    The MASS LIGHT DARK PIPELINE (with lens light) uses one search to fits a complex lens mass model to a high level of 
    accuracy, using the source model of the SOURCE PIPELINE and the lens light model of the LIGHT LP PIPELINE to 
    initialize the model priors . In this example it:
    
     - Uses a parametric `Sersic` bulge and `Sersic` disk with centres aligned for the lens galaxy's 
     light and its stellar mass [12 parameters: fixed from LIGHT LP PIPELINE].
    
     - The lens galaxy's dark matter mass distribution is a `NFWMCRLudlow` whose centre is aligned with bulge of 
     the light and stellar mass model above [5 parameters].
    
     - Uses an `Inversion` for the source's light [priors fixed from SOURCE PIX PIPELINE].
    
     - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the MASS 
     LIGHT DARK PIPELINE.
    """
    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_pix_result_2.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
        use_jax=True,
    )

    lp_chain_tracer = al.util.chaining.lp_chain_tracer_from(
        light_result=light_result, settings_search=settings_search
    )

    dark = af.Model(al.mp.NFWMCRLudlow)

    mass_result = slam_pipeline.mass_light_dark.run(
        settings_search=settings_search,
        analysis=analysis,
        lp_chain_tracer=lp_chain_tracer,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        light_result=light_result,
        dark=dark,
    )

    dark = af.Model(al.mp.NFWMCRLudlowSph)

    mass_result = slam_pipeline.mass_light_dark.run(
        settings_search=settings_search,
        analysis=analysis,
        lp_chain_tracer=lp_chain_tracer,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        light_result=light_result,
        dark=dark,
    )

    dark = af.Model(al.mp.NFWMCRLudlow)

    mass_result = slam_pipeline.mass_light_dark.run(
        settings_search=settings_search,
        analysis=analysis,
        lp_chain_tracer=lp_chain_tracer,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        light_result=light_result,
        dark=None,
    )


if __name__ == "__main__":
    fit()
