from torchvision.io import decode_image
from torchvision.models import get_model, get_model_weights
from torchcam.methods import LayerCAM
from torchvision.transforms.v2.functional import to_pil_image
from torchcam.utils import overlay_mask
from torch.nn.functional import softmax
from torch import topk
import matplotlib.pyplot as plt
import os


class ResnetVisualizer:
    def __init__(self):
        self.weights = get_model_weights("resnet18").DEFAULT
        self.model = get_model("resnet18", weights=self.weights).eval()
        self.preprocess = self.weights.transforms()
        self.labels = self.weights.meta["categories"]
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}

        self.layers = {
            "1": self.model.layer1,
            "2": self.model.layer2,
            "3": self.model.layer3,
            "4": self.model.layer4
        }

        self.img = None
        self.logits = None
        self.top_k_index = None
        self.top_k_probs = None
        self.target_activation_map = None
        self.top1_activation_map = None
        self.layer = "4"

    def fit_image(self, img_path, target_class, top_k=5, layer=4):
        self.layer = str(layer)
        target_layer = self.layers[self.layer]

        self.img = decode_image(img_path)
        img_tensor = self.preprocess(self.img)

        target_class_id = self.label_to_id[target_class]

        with LayerCAM(self.model, target_layer=target_layer) as cam_extractor:
            self.logits = self.model(img_tensor.unsqueeze(0))

            probs = softmax(self.logits, dim=1)
            top_probs, top_ids = topk(probs, top_k)

            self.top_k_index = top_ids[0]
            self.top_k_probs = top_probs[0]

            self.target_activation_map = cam_extractor(
                target_class_id,
                self.logits
            )[0].squeeze(0)

            top1_id = self.top_k_index[0].item()
            logits_top1 = self.model(img_tensor.unsqueeze(0))
            self.top1_activation_map = cam_extractor(
                top1_id,
                logits_top1
            )[0].squeeze(0)

    def visualize(self, target_class, example_type, show_top1=True):
        img_pil = to_pil_image(self.img)

        target_overlay = overlay_mask(
            img_pil,
            to_pil_image(self.target_activation_map, mode="F"),
            alpha=0.5
        )

        if show_top1:
            top1_overlay = overlay_mask(
                img_pil,
                to_pil_image(self.top1_activation_map, mode="F"),
                alpha=0.5
            )

            top1_id = self.top_k_index[0].item()
            top1_name = self.labels[top1_id]
            top1_prob = self.top_k_probs[0].item()

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))

            axes[0].imshow(img_pil)
            axes[0].set_title("Originalbild")
            axes[0].axis("off")

            axes[1].imshow(target_overlay)
            axes[1].set_title(f"CAM för målklass\n{target_class} ({example_type})")
            axes[1].axis("off")

            axes[2].imshow(top1_overlay)
            axes[2].set_title(f"CAM för top-1\n{top1_name} ({top1_prob:.3f})")
            axes[2].axis("off")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            axes[0].imshow(img_pil)
            axes[0].set_title("Originalbild")
            axes[0].axis("off")

            axes[1].imshow(target_overlay)
            axes[1].set_title(f"CAM för målklass\n{target_class} ({example_type})")
            axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    def print_topk(self, top_k=5):
        print("Top-5 prediktioner:")
        for i in range(min(top_k, len(self.top_k_index))):
            class_id = self.top_k_index[i].item()
            prob = self.top_k_probs[i].item()
            print(f"{i+1}. {self.labels[class_id]} ({prob:.4f})")

    def print_logit_analysis(self, top_k=10):
        print("Logit-analys:")
        values, ids = topk(self.logits[0], top_k)

        for i, (value, class_id) in enumerate(zip(values, ids), start=1):
            print(f"{i}. {self.labels[class_id.item()]} | logit = {value.item():.4f}")

    def fit_visualize_image(self, img_path, target_class, example_type, top_k=5, layer=4, show_top1=True):
        if not os.path.exists(img_path):
            print(f"Bild saknas: {img_path}")
            return

        self.fit_image(
            img_path=img_path,
            target_class=target_class,
            top_k=top_k,
            layer=layer
        )

        self.visualize(
            target_class=target_class,
            example_type=example_type,
            show_top1=show_top1
        )

        self.print_topk(top_k=top_k)

        if example_type == "positive":
            print(
                f"Tolkning: Detta är en positiv bild för {target_class}. "
                f"CAM för målklassen bör främst markera objektet."
            )
        else:
            print(
                f"Tolkning: Detta är en negativ bild för {target_class}. "
                f"CAM kan vara svagare eller mer utspridd eftersom målklassen inte finns i bilden."
            )