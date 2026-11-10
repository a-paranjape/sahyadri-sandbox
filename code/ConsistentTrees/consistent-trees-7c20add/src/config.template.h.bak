//string(a,b) real(a,b) integer(a,b)
//Sets strings, floats, and integers; 
//     "a" is the variable name; 
//     "b" is the default.

real(Om,0.27);
real(h0,0.70);
real(Ol,0.73);

string(MASS_DEFINITION, "vir");

string(SCALEFILE,"scales.txt");
string(INBASE,".");
string(OUTBASE,".");
string(TREE_OUTBASE,".");
string(HLIST_OUTBASE,".");

string(INPUT_FORMAT, "ASCII");

real(MAJOR_MERGER,0.3);
real(MIN_MMP_MASS_RATIO,0.5);
real(MIN_MMP_VMAX_RATIO,0.7);

real(BOX_WIDTH , 0);
real(BOX_DIVISIONS , 5);

real(SOFTENING_LENGTH , 0.2);

integer(PADDING_TIMESTEPS,0);
integer(MIN_TIMESTEPS_SUB_TRACKED , 20); // Only keep tracks if they have been around for longer than this time.
integer(MIN_TIMESTEPS_SUB_MMP_TRACKED , 10); // Only keep tracks if they have been around for longer than this time.
integer(MIN_TIMESTEPS_TRACKED , 5); // Only keep tracks if they have been around for longer than this time.
real(MAX_PHANTOM_FRACTION , 0.2); // Discard tracks where there are more than 20% phantoms.

real(LAST_DITCH_SEARCH_LIMIT,1.0); /* For connecting halos which have "moved" up to this amount
				       times their virial radius */
real(LAST_DITCH_VMAX_RATIO_1,1.1); /* For connecting halos which have "moved" unphysical amounts */
real(LAST_DITCH_VMAX_RATIO_2,2.5); /* For connecting halos which have "moved" unphysical amounts */

integer(MAX_PHANTOM,4); /* max timesteps to keep phantom halo */
integer(MAX_PHANTOM_SMALL,2); /* max timesteps to keep small phantom halo */
integer(SMALL_PARTICLE_LIMIT,49); /* Halos smaller than this size get kept around for less time. */
/* SHOULD NOT BE SET TO MORE THAN 50 for BOLSHOI and BDM results. */

real(TIDAL_FORCE_LIMIT,0.1);
real(TIDAL_FORCE_CALCULATION_LIMIT, 1e-3);
integer(RECURSION_LIMIT,5);
real(METRIC_LIMIT,7);
real(UNPHYSICAL,22); //Should be (3*METRIC_LIMIT+1));
real(METRIC_BREAK_LIMIT,3.2); //Below which we break a link.
real(MASS_RES_OK,1e11); //Halo mass above which there are probably not resolution issues.

integer(EXTRA_PARAMS, 0);                 //Number of extra parameters
string(EXTRA_PARAM_LABELS, "");           //Labels for output
string(EXTRA_PARAM_DESCRIPTIONS, "");     //Descriptions in merger tree, etc.
string(EXTRA_PARAM_INTERPOLATIONS, "");   //Interpolation method: "l", "s", "c" for linear, square, cubic
integer(SUSSING_MASS_FIELD, -1);
integer(FIX_ROCKSTAR_SPINS, -1);
integer(RESCALE_RVIR, 0);

integer(LIMITED_MEMORY, 0);
