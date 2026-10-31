#ifndef MODEL_API_H
#define MODEL_API_H

/* Region (1..4) for one feature vector in model input order (see include/model.h). */
int hif_predict(const float *features);

/* HIF_N_FEATURES of the model.h this was built with, for a consistency check in main.c. */
extern const int hif_model_n_features;

#endif
