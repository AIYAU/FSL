import torch
from torch import Tensor, nn

from .few_shot_classifier import FewShotClassifier


class TIM(FewShotClassifier):
    """
    Malik Boudiaf, Ziko Imtiaz Masud, Jérôme Rony, José Dolz, Pablo Piantanida, Ismail Ben Ayed.
    "Transductive Information Maximization For Few-Shot Learning" (NeurIPS 2020)
    https://arxiv.org/abs/2008.11297

    Fine-tune prototypes based on
        1) classification error on support images
        2) mutual information between query features and their label predictions
    Classify w.r.t. to euclidean distance to updated prototypes.
    As is, it is incompatible with episodic training because we freeze the backbone to perform
    fine-tuning.

    TIM is a transductive method.
    """

    def __init__(
        self,
        *args,
        fine_tuning_steps: int = 50,
        fine_tuning_lr: float = 1e-4,
        cross_entropy_weight: float = 1.0,
        marginal_entropy_weight: float = 1.0,
        conditional_entropy_weight: float = 0.1,
        temperature: float = 10.0,
        **kwargs,
    ):
        """
        Args:
            fine_tuning_steps: number of fine-tuning steps
            fine_tuning_lr: learning rate for fine-tuning
            cross_entropy_weight: weight given to the cross-entropy term of the loss
            marginal_entropy_weight: weight given to the marginal entropy term of the loss
            conditional_entropy_weight: weight given to the conditional entropy term of the loss
            temperature: temperature applied to the logits before computing
                softmax or cross-entropy. Higher temperature means softer predictions.
        """
        super().__init__(*args, **kwargs)

        # Since we fine-tune the prototypes we need to make them leaf variables
        # i.e. we need to freeze the backbone.
        self.backbone.requires_grad_(False)

        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr
        self.cross_entropy_weight = cross_entropy_weight
        self.marginal_entropy_weight = marginal_entropy_weight
        self.conditional_entropy_weight = conditional_entropy_weight
        self.temperature = temperature

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Fine-tune prototypes based on support classification error and mutual information between
        query features and their label predictions.
        Then classify w.r.t. to euclidean distance to prototypes.
        """
        query_features = self.compute_features(query_images)

        num_classes = self.support_labels.unique().size(0)
        support_labels_one_hot = nn.functional.one_hot(  # pylint: disable=not-callable
            self.support_labels, num_classes
        )

        with torch.enable_grad():
            self.prototypes.requires_grad_()
            optimizer = torch.optim.Adam([self.prototypes], lr=self.fine_tuning_lr)

            for _ in range(self.fine_tuning_steps):
                support_logits = self.temperature * self.cosine_distance_to_prototypes(
                    self.support_features
                )
                query_logits = self.temperature * self.cosine_distance_to_prototypes(
                    query_features
                )

                support_cross_entropy = (
                    -(support_labels_one_hot * support_logits.log_softmax(1))
                    .sum(1)
                    .mean(0)
                )

                query_soft_probs = query_logits.softmax(1)
                query_conditional_entropy = (
                    -(query_soft_probs * torch.log(query_soft_probs + 1e-12))
                    .sum(1)
                    .mean(0)
                )

                marginal_prediction = query_soft_probs.mean(0)
                marginal_entropy = -(
                    marginal_prediction * torch.log(marginal_prediction)
                ).sum(0)

                loss = self.cross_entropy_weight * support_cross_entropy - (
                    self.marginal_entropy_weight * marginal_entropy
                    - self.conditional_entropy_weight * query_conditional_entropy
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.softmax_if_specified(
            self.cosine_distance_to_prototypes(query_features),
            temperature=self.temperature,
        ).detach()

    @staticmethod
    def is_transductive() -> bool:
        return True
