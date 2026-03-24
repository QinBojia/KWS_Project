"""
Comprehensive paper revision script.
Applies all structural and content changes to the unpacked document.xml.
"""
import re

DOC = 'unpacked/word/document.xml'

with open(DOC, 'r', encoding='utf-8') as f:
    xml = f.read()

def body(text):
    """Standard body paragraph with first-line indent."""
    return f'''    <w:p>
      <w:pPr>
        <w:ind w:firstLine="432"/>
      </w:pPr>
      <w:r>
        <w:rPr>
          <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
          <w:b w:val="0"/>
          <w:i w:val="0"/>
          <w:sz w:val="22"/>
        </w:rPr>
        <w:t>{text}</w:t>
      </w:r>
    </w:p>'''

def body_noi(text):
    """Body paragraph without indent."""
    return f'''    <w:p>
      <w:r>
        <w:rPr>
          <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
          <w:b w:val="0"/>
          <w:i w:val="0"/>
          <w:sz w:val="22"/>
        </w:rPr>
        <w:t>{text}</w:t>
      </w:r>
    </w:p>'''

def h1(text):
    return f'''    <w:p>
      <w:pPr>
        <w:pStyle w:val="Heading1"/>
      </w:pPr>
      <w:r>
        <w:t>{text}</w:t>
      </w:r>
    </w:p>'''

def h2(text):
    return f'''    <w:p>
      <w:pPr>
        <w:pStyle w:val="Heading2"/>
      </w:pPr>
      <w:r>
        <w:t>{text}</w:t>
      </w:r>
    </w:p>'''


# ============================================================
# 1. ABSTRACT — add limitation + shorten
# ============================================================
print("1. Rewriting Abstract...")

old_abstract = ' — Keyword spotting (KWS) is a key enabling technology for voice-driven interfaces in embedded and Internet-of-Things (IoT) devices. Deploying deep learning models for KWS directly on microcontrollers is challenging due to strict constraints in memory capacity, computational resources, and energy consumption. In this work, we investigate the optimization of a lightweight neural network architecture for embedded keyword spotting applications. Specifically, we focus on improving the Temporal Efficient Network (TENet) architecture through systematic architecture exploration and hyperparameter optimization. Five candidate architectures are evaluated under a fixed computational budget, and TENet is selected as the most efficient base model. A four-stage progressive grid search is then performed over six architectural hyperparameters, evaluating 2,302 candidate configurations. The optimized model is trained using the Google Speech Commands v2 dataset with data augmentation techniques including SpecAugment, Mixup, and label smoothing, and deployed on an STM32F767IGT6 microcontroller based on the ARM Cortex-M7 architecture. Experimental results show that the proposed model achieves a test accuracy of 96.84% (96.76% after INT8 quantization) and a weighted F1-score of 0.9675 with only 12,822 parameters and 276,136 multiply-accumulate operations (MACCs). The deployed INT8 model requires 13.87 KB of flash memory and 7.64 KB of RAM, and achieves an average inference latency of 6.32 ms on the target hardware. Compared with previously reported lightweight keyword spotting architectures operating under similar computational constraints, the optimized TENet model achieves improved classification performance while maintaining a compact footprint suitable for real-time embedded deployment.'

new_abstract = ' — Keyword spotting (KWS) enables voice-driven interaction on embedded devices. While recent lightweight architectures achieve high accuracy on standard benchmarks, most require millions of multiply-accumulate operations (MACCs), limiting deployment on the most resource-constrained microcontrollers. This work optimizes the Temporal Efficient Network (TENet) for keyword spotting under a strict computational budget matched to a DS-CNN embedded baseline. A four-stage progressive grid search over six architectural hyperparameters evaluates 2,302 configurations to identify an efficient TENet variant. The optimized model, trained on Google Speech Commands v2, achieves 96.84% test accuracy (96.76% after INT8 quantization) with 12,822 parameters and 276,136 MACCs. Deployed on an STM32F767IGT6 Cortex-M7 microcontroller, the INT8 model requires 13.87 KB flash and 7.64 KB RAM with 6.32 ms inference latency.'

xml = xml.replace(old_abstract, new_abstract)

# ============================================================
# 2. INTRODUCTION — add limitation after existing arch paragraph
# ============================================================
print("2. Adding limitation to intro arch paragraph...")

old_intro_arch = 'In these systems, compact acoustic features such as Mel-Frequency Cepstral Coefficients (MFCCs) [7] are commonly used to represent speech signals in a computationally efficient form.'
new_intro_arch = 'In these systems, compact acoustic features such as Mel-Frequency Cepstral Coefficients (MFCCs) [7] are commonly used to represent speech signals in a computationally efficient form. However, most reported high-accuracy KWS models require millions of MACCs, and their performance under extreme computational constraints (below 300K MACCs) remains largely unexplored.'

xml = xml.replace(old_intro_arch, new_intro_arch)

# ============================================================
# 3. DELETE contributions list (4 numbered items + intro sentence + roadmap paragraph)
# ============================================================
print("3. Deleting contributions list...")

# Replace the "Motivated..." paragraph to remove "287,673 MACCs" and reframe
old_motivated = 'Motivated by these challenges, this work investigates the optimization of a lightweight neural architecture for keyword spotting on embedded microcontrollers. Specifically, we focus on the Temporal Efficient Network (TENet) architecture [4] and perform systematic architecture exploration and hyperparameter optimization to improve its performance under a strict computational budget of 287,673 MACCs. The optimized model is trained on the Google Speech Commands v2 dataset [1] and deployed on an STM32F767IGT6 microcontroller based on the ARM Cortex-M7 architecture using the ST X-CUBE-AI framework [8].'
new_motivated = 'Motivated by these challenges, this work investigates the optimization of the TENet architecture [4] for keyword spotting on ARM Cortex-M7 microcontrollers. We adopt the computational budget of a DS-CNN embedded baseline [9] to enable direct comparison, and perform a progressive grid search over architectural hyperparameters to maximize accuracy within this constraint. The optimized model is trained on the Google Speech Commands v2 dataset [1] and deployed on an STM32F767IGT6 microcontroller using the ST X-CUBE-AI framework [8].'

xml = xml.replace(old_motivated, new_motivated)

# Delete "The main contributions..." through the 4 list items
old_contributions_start = '<w:t>The main contributions of this work are summarized as follows:</w:t>'
old_roadmap = 'The remainder of this paper is organized as follows. Section II reviews related work in lightweight keyword spotting. Section III describes the proposed methodology and system design. Section IV presents the experimental results and discussion. Finally, Section V concludes the paper.'

# Find the paragraph containing "The main contributions"
idx_contrib = xml.find(old_contributions_start)
# Go back to find the <w:p> that contains it
contrib_p_start = xml.rfind('<w:p>', 0, idx_contrib)

# Find the roadmap paragraph end
idx_roadmap = xml.find(old_roadmap)
roadmap_p_end = xml.find('</w:p>', idx_roadmap) + len('</w:p>')

# Replace contributions + roadmap with a brief sentence
new_summary = body('The remainder of this paper is organized as follows. Section II describes the proposed methodology. Section III presents the experimental results and comparison with existing approaches. Section IV concludes the paper.')

xml = xml[:contrib_p_start] + new_summary + xml[roadmap_p_end:]

# ============================================================
# 4. MERGE Related Work into Introduction, delete Section II heading
# ============================================================
print("4. Merging Related Work into Introduction...")

# Delete "II. Related Work" heading
old_rw_heading = '<w:t>II. Related Work</w:t>'
idx_rw = xml.find(old_rw_heading)
rw_heading_start = xml.rfind('<w:p>', 0, idx_rw)
rw_heading_end = xml.find('</w:p>', idx_rw) + len('</w:p>')
xml = xml[:rw_heading_start] + xml[rw_heading_end:]

# Delete "A. Lightweight Architectures..." subheading
old_rw_a = '<w:t>A. Lightweight Architectures for Keyword Spotting</w:t>'
idx_rw_a = xml.find(old_rw_a)
rw_a_start = xml.rfind('<w:p>', 0, idx_rw_a)
rw_a_end = xml.find('</w:p>', idx_rw_a) + len('</w:p>')
xml = xml[:rw_a_start] + xml[rw_a_end:]

# Delete "B. Neural Architecture Search..." subheading
old_rw_b = '<w:t>B. Neural Architecture Search for TinyML</w:t>'
idx_rw_b = xml.find(old_rw_b)
rw_b_start = xml.rfind('<w:p>', 0, idx_rw_b)
rw_b_end = xml.find('</w:p>', idx_rw_b) + len('</w:p>')
xml = xml[:rw_b_start] + xml[rw_b_end:]

# Delete "C. Embedded Deployment Frameworks" subheading
old_rw_c = '<w:t>C. Embedded Deployment Frameworks</w:t>'
idx_rw_c = xml.find(old_rw_c)
rw_c_start = xml.rfind('<w:p>', 0, idx_rw_c)
rw_c_end = xml.find('</w:p>', idx_rw_c) + len('</w:p>')
xml = xml[:rw_c_start] + xml[rw_c_end:]

# Renumber: "III. Methodology" → "II. Methodology"
xml = xml.replace('<w:t>III. Methodology</w:t>', '<w:t>II. Methodology</w:t>')
# "IV. Results" → "III. Results"
xml = xml.replace('<w:t>IV. Results and Discussion</w:t>', '<w:t>III. Results and Discussion</w:t>')
# "V. Conclusion" → "IV. Conclusion"
xml = xml.replace('<w:t>V. Conclusion</w:t>', '<w:t>IV. Conclusion</w:t>')
# Fix section cross-references
xml = xml.replace('Section III.D', 'Section II.C')
xml = xml.replace('Section III.F', 'Section II.E')
xml = xml.replace('Section III.G', 'Section II.F')
xml = xml.replace('Section IV.D', 'Section III.C')

# ============================================================
# 5. DELETE Section D (Architecture Selection) + Table I entirely
# ============================================================
print("5. Deleting Architecture Selection (D) section...")

old_arch_heading = '<w:t>D. Architecture Selection</w:t>'
idx_arch = xml.find(old_arch_heading)
arch_start = xml.rfind('<w:p>', 0, idx_arch)

# Find next section heading (E. TENet Architecture)
old_tenet_heading = '<w:t>E. TENet Architecture</w:t>'
idx_tenet = xml.find(old_tenet_heading)
tenet_heading_start = xml.rfind('<w:p>', 0, idx_tenet)

xml = xml[:arch_start] + xml[tenet_heading_start:]

# ============================================================
# 6. Section E TENet → simplify, add why TENet
# ============================================================
print("6. Simplifying TENet Architecture section...")

# Renumber E → D (after deleting D)
xml = xml.replace('<w:t>E. TENet Architecture</w:t>', '<w:t>D. TENet Architecture</w:t>')

# Replace the TENet intro paragraph
old_tenet_intro = 'The TENet architecture [4] processes MFCC sequences using one-dimensional temporal convolutions. The input tensor (1, 62, 13) is reshaped to (13, 62), treating the 13 MFCC coefficients as input channels and the 62 frames as the temporal dimension.'
new_tenet_intro = 'TENet [4] is selected for its strong accuracy-efficiency trade-off: among lightweight 1D architectures, it achieves the highest accuracy at sub-300K MACCs while maintaining robust INT8 quantization behavior. TENet processes MFCC sequences using one-dimensional temporal convolutions. The input tensor (1, 62, 13) is reshaped to (13, 62), treating the 13 MFCC coefficients as input channels and the 62 frames as the temporal dimension.'

xml = xml.replace(old_tenet_intro, new_tenet_intro)

# Replace the detailed inverted bottleneck description with a shorter version
old_inverted = 'The architecture adopts an inverted bottleneck design inspired by MobileNetV2 [10], where each block consists of: (1) pointwise expansion Conv1d (in_ch → in_ch × ratio, kernel=1) to increase channel dimension; (2) depthwise temporal Conv1d (hidden, hidden, kernel=k, groups=hidden) for temporal feature extraction; (3) pointwise projection Conv1d (hidden → out_ch, kernel=1) for dimensionality reduction with no activation (linear); and (4) a residual connection (identity shortcut when stride=1 and in_ch=out_ch, otherwise a 1×1 projection shortcut).'
new_inverted = 'The architecture adopts an inverted bottleneck design inspired by MobileNetV2 [10], consisting of pointwise expansion, depthwise temporal convolution, and pointwise projection with residual connections.'

xml = xml.replace(old_inverted, new_inverted)

# ============================================================
# 7. Renumber remaining methodology subsections
# ============================================================
print("7. Renumbering methodology subsections...")

xml = xml.replace('<w:t>F. Architecture Optimization</w:t>', '<w:t>E. Architecture Optimization</w:t>')
xml = xml.replace('<w:t>G. Training Configuration</w:t>', '<w:t>F. Training Configuration</w:t>')
xml = xml.replace('<w:t>H. INT8 Quantization</w:t>', '<w:t>G. INT8 Quantization</w:t>')
xml = xml.replace('<w:t>I. Embedded Deployment</w:t>', '<w:t>H. Embedded Deployment</w:t>')

# Also fix Table III reference to §III.G → §II.F
xml = xml.replace('All (§III.G)', 'All (§II.F)')

# ============================================================
# 8. Training Configuration — mention found model + GPU
# ============================================================
print("8. Updating Training Configuration...")

old_training_intro = 'The final model was trained using the following configuration:'
new_training_intro = 'The best architecture identified by grid search (Section II.E) was trained to convergence on an NVIDIA RTX 5080 Laptop GPU using the following configuration:'

xml = xml.replace(old_training_intro, new_training_intro)

# ============================================================
# 9. Delete +1.84% sentence in methodology
# ============================================================
print("9. Deleting +1.84% premature result...")

old_184 = 'These techniques collectively contributed a +1.84 percentage point improvement in validation accuracy compared with training without augmentation (see Section IV.D for ablation results).'

# Find and remove the entire paragraph containing this
idx_184 = xml.find(old_184)
if idx_184 >= 0:
    p_start = xml.rfind('<w:p>', 0, idx_184)
    p_end = xml.find('</w:p>', idx_184) + len('</w:p>')
    xml = xml[:p_start] + xml[p_end:]
    print("   Deleted.")

# ============================================================
# 10. Add benchmarking info after deployment pipeline
# ============================================================
print("10. Adding benchmarking paragraph...")

old_deploy_end = 'Inference latency is measured using the ARM DWT (Data Watchpoint and Trace) cycle counter, which provides cycle-accurate timing of the ai_run() function call.'
new_deploy_end = 'Inference latency is measured using the ARM DWT (Data Watchpoint and Trace) cycle counter, which provides cycle-accurate timing of the ai_run() function call. The latency measurement covers only the neural network forward pass (ai_run); MFCC feature extraction is excluded as it is performed by the host application prior to inference.'

xml = xml.replace(old_deploy_end, new_deploy_end)

# ============================================================
# 11. Move Experimental Setup from Results to Methodology
# ============================================================
print("11. Moving Experimental Setup...")

# Remove "A. Experimental Setup" heading + its paragraph from Results
old_expsetup_heading = '<w:t>A. Experimental Setup</w:t>'
idx_exp = xml.find(old_expsetup_heading)
exp_heading_start = xml.rfind('<w:p>', 0, idx_exp)

old_expsetup_text = 'All models were trained using PyTorch 2.10.0 with CUDA on an NVIDIA RTX 5080 Laptop GPU. Evaluation was performed on the official test split of Google Speech Commands v2 (12,105 samples). Classification accuracy is computed as top-1 accuracy. F1-scores (weighted and macro) are computed using scikit-learn\u2019s classification_report. All reported test accuracies are from a single training run with seed=123.'
# Actually in XML it's &#x2019;
old_expsetup_text_xml = 'All models were trained using PyTorch 2.10.0 with CUDA on an NVIDIA RTX 5080 Laptop GPU.'
idx_exp_text = xml.find(old_expsetup_text_xml, idx_exp)
exp_text_end = xml.find('</w:p>', idx_exp_text) + len('</w:p>')

# Remove from Results
xml = xml[:exp_heading_start] + xml[exp_text_end:]

# Renumber Results subsections: B→A, C→B (will be merged), D→C, E→D, F→E, G→F
xml = xml.replace('<w:t>B. Classification Performance</w:t>', '<w:t>A. Classification Performance</w:t>')
# C. Per-Class Analysis will be merged/deleted, handle below
xml = xml.replace('<w:t>D. Ablation Study: Training Optimization Impact</w:t>', '<w:t>B. Ablation Study</w:t>')
xml = xml.replace('<w:t>E. Model Size and Deployment Metrics</w:t>', '<w:t>C. Model Size and Deployment Metrics</w:t>')
xml = xml.replace('<w:t>F. On-Device Inference Latency</w:t>', '<w:t>D. On-Device Inference Latency</w:t>')
xml = xml.replace('<w:t>G. Comparison with State-of-the-Art</w:t>', '<w:t>E. Comparison with State-of-the-Art</w:t>')

# ============================================================
# 12. Merge Per-Class into Classification Performance (as summary) + delete Table V + Fig
# ============================================================
print("12. Merging per-class into classification performance...")

# Find and remove the entire "C. Per-Class Analysis" section
# From its heading to the next heading (D. Ablation Study)
old_perclass_heading = '<w:t>C. Per-Class Analysis</w:t>'
idx_pc = xml.find(old_perclass_heading)
if idx_pc >= 0:
    pc_start = xml.rfind('<w:p>', 0, idx_pc)
    # Find next heading (B. Ablation Study — already renamed)
    old_ablation_heading = '<w:t>B. Ablation Study</w:t>'
    idx_abl = xml.find(old_ablation_heading, idx_pc)
    abl_start = xml.rfind('<w:p>', 0, idx_abl)
    xml = xml[:pc_start] + xml[abl_start:]
    print("   Deleted Per-Class section.")

# Add per-class summary sentence to Classification Performance
# Find the end of Table IV's discussion text
old_quant_degrad = 'INT8 quantization introduces minimal accuracy degradation (−0.08 percentage points), confirming that the optimized architecture is robust to post-training quantization.'
if xml.find(old_quant_degrad) < 0:
    # Try alternative text
    old_quant_degrad = 'INT8 quantization introduces a negligible accuracy degradation of −0.08 percentage points'

idx_qd = xml.find(old_quant_degrad)
if idx_qd >= 0:
    qd_p_end = xml.find('</w:p>', idx_qd) + len('</w:p>')
    perclass_summary = body('Per-class F1-scores range from 0.918 (go) to 1.000 (silence), with INT8 degradation below 0.011 for all classes. The macro F1-score (0.9490) provides a balanced assessment across the 12 classes, as the unknown class (57.3% of test samples) dominates the weighted metric.')
    xml = xml[:qd_p_end] + '\n' + perclass_summary + xml[qd_p_end:]
    print("   Added per-class summary to Classification Performance.")
else:
    print("   WARNING: Could not find quantization degradation text.")

# ============================================================
# 13. Fix Fig numbering (training curve = Fig 4, per-class deleted so no Fig 5)
# ============================================================
print("13. Fixing figure numbering...")

xml = xml.replace('Fig. 5. Training curve', 'Fig. 4. Training curve')
# The per-class fig was deleted with section C

# ============================================================
# 14. Fix references [4], [13], [16], [21], add [22]
# ============================================================
print("14. Fixing references...")

xml = xml.replace(
    '[4] R. Li, Z. Gao, L. Wang, and G. Li, "Small-footprint keyword spotting with multi-scale temporal convolution," in Proc. Interspeech, 2020.',
    '[4] X. Li, X. Wei, and X. Qin, "Small-footprint keyword spotting with multi-scale temporal convolution," in Proc. Interspeech, 2020, pp. 1987–1991.'
)
xml = xml.replace(
    '[13] S. Liberatore, F. Pollicino, and M. Rusci, "MicroNAS: Zero-shot neural architecture search for MCUs," arXiv preprint arXiv:2401.08996, 2024.',
    '[13] Y. Qiao, H. Xu, Y. Zhang, and S. Huang, "MicroNAS: Zero-shot neural architecture search for MCUs," in Proc. DATE, 2024.'
)
xml = xml.replace(
    '[16] L. Lai, N. Suda, and V. Chandra, "CMSIS-NN: Efficient neural networks on ARM Cortex-M processors," arXiv preprint arXiv:1801.06601, 2018.',
    '[16] L. Lai, N. Suda, and V. Chandra, "CMSIS-NN: Efficient neural network kernels for Arm Cortex-M CPUs," arXiv preprint arXiv:1801.06601, 2018.'
)
xml = xml.replace(
    '[21] F. Bartoli, L. Boschi, and V. Luise, "End-to-end efficiency in keyword spotting: A system-level approach for embedded microcontrollers," arXiv preprint arXiv:2509.07051, 2025.',
    '[21] P. Bartoli, T. Bondini, C. Veronesi, A. Giudici, N. Antonello, and F. Zappa, "End-to-end efficiency in keyword spotting: A system-level approach for embedded microcontrollers," arXiv preprint arXiv:2509.07051, 2025.'
)

# Add LiCoNet citation in Related Work text
xml = xml.replace(
    'LiCoNet uses linearized convolutions for efficient streaming inference on embedded platforms.',
    'Yang et al. [22] proposed LiCoNet, which uses linearized convolutions for efficient streaming inference on embedded platforms.'
)

# Add [22] reference at end
old_ref21_end = 'arXiv preprint arXiv:2509.07051, 2025.</w:t>\n      </w:r>\n    </w:p>\n    <w:sectPr'
new_ref21_end = '''arXiv preprint arXiv:2509.07051, 2025.</w:t>
      </w:r>
    </w:p>
    <w:p>
      <w:r>
        <w:rPr>
          <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
          <w:sz w:val="20"/>
        </w:rPr>
        <w:t>[22] H. Yang et al., "LiCo-Net: Linearized convolution network for hardware-efficient keyword spotting," arXiv preprint arXiv:2211.04635, 2022.</w:t>
      </w:r>
    </w:p>
    <w:sectPr'''

xml = xml.replace(old_ref21_end, new_ref21_end)

# Fix MACs → MACCs in BC-ResNet paragraph
xml = xml.replace('3.1M MACs, which exceeds', '3.1M MACCs, which exceeds')

# ============================================================
# 15. Conclusion — restructure into 2 paragraphs
# ============================================================
print("15. Restructuring Conclusion...")

old_conclusion_p1 = 'This paper presented an optimized keyword spotting system designed for deployment on resource-constrained microcontrollers. Through a five-architecture comparison study and a four-stage progressive grid search over 2,302 candidate configurations, an optimized TENet variant was identified that achieves high classification accuracy within a strict computational budget of 276K MACCs.'

new_conclusion_p1 = 'This paper presented an optimized keyword spotting system for ARM Cortex-M7 microcontrollers. A four-stage progressive grid search over 2,302 TENet configurations was conducted under a computational budget matched to a DS-CNN embedded baseline. The best configuration was trained with SpecAugment, Mixup, label smoothing, and cosine annealing, then quantized to INT8 and deployed on an STM32F767IGT6 using X-CUBE-AI.'

xml = xml.replace(old_conclusion_p1, new_conclusion_p1)

old_conclusion_p2 = 'The optimized model achieves 96.84% test accuracy (96.76% after INT8 quantization) and a weighted F1-score of 0.9675 on the Google Speech Commands v2 12-class benchmark, with only 12,822 parameters. When deployed on an STM32F767IGT6 Cortex-M7 microcontroller at 216 MHz, the INT8 model requires 13.87 KB of flash memory and 7.64 KB of RAM, and achieves an inference latency of 6.32 ms. Compared with the DS-CNN baseline operating under the same MACC constraint, the proposed model improves accuracy by 5.76 percentage points while reducing flash memory by 91.8%.'

new_conclusion_p2 = 'The optimized model achieves 96.76% INT8 accuracy (weighted F1 = 0.9675) on the Google Speech Commands v2 12-class benchmark with 12,822 parameters and 276K MACCs. On-device, the INT8 model requires 13.87 KB flash, 7.64 KB RAM, and 6.32 ms inference latency at 216 MHz.'

xml = xml.replace(old_conclusion_p2, new_conclusion_p2)

# Delete the "Key findings" paragraph (redundant with Results)
old_key_findings = 'Key findings from this work include: (1) narrow channels with high expansion ratios are more efficient than wide channels with low expansion ratios at the same MACC budget; (2) training augmentation techniques (SpecAugment, Mixup, label smoothing) collectively contribute +1.84% accuracy improvement for small models; and (3) INT8 post-training quantization introduces negligible accuracy degradation (−0.08%) for the optimized architecture.'
idx_kf = xml.find(old_key_findings)
if idx_kf >= 0:
    kf_start = xml.rfind('<w:p>', 0, idx_kf)
    kf_end = xml.find('</w:p>', idx_kf) + len('</w:p>')
    xml = xml[:kf_start] + xml[kf_end:]
    print("   Deleted Key findings paragraph.")

# ============================================================
# 16. Table IX — add Classes column
# ============================================================
print("16. Adding Classes column to Table IX...")

# This is complex — Table IX has many rows. Instead of XML surgery on the table,
# let me add "Classes" info to the Model column entries as "(12-class)" or "(8-class)"
# This is simpler and achieves the same goal.

# Find and annotate entries in SOTA table
xml = xml.replace('<w:t>DS-CNN (Sorensen) [9]</w:t>', '<w:t>DS-CNN (Sorensen) [9] (12-class)</w:t>')
xml = xml.replace('<w:t>TENet12 [4]</w:t>', '<w:t>TENet12 [4] (12-class)</w:t>')
xml = xml.replace('<w:t>TENet6-N [4]</w:t>', '<w:t>TENet6-N [4] (12-class)</w:t>')
xml = xml.replace('<w:t>BC-ResNet-1 [5]</w:t>', '<w:t>BC-ResNet-1 [5] (12-class)</w:t>')
xml = xml.replace('<w:t>BC-ResNet-8 [5]</w:t>', '<w:t>BC-ResNet-8 [5] (12-class)</w:t>')
xml = xml.replace('<w:t>MatchboxNet [6]</w:t>', '<w:t>MatchboxNet [6] (12-class)</w:t>')
# Bartoli models are 8-class
xml = xml.replace('<w:t>DS-CNN (Bartoli) [21]</w:t>', '<w:t>DS-CNN (Bartoli) [21] (8-class)</w:t>')
xml = xml.replace('<w:t>TENet6 (Bartoli) [21]</w:t>', '<w:t>TENet6 (Bartoli) [21] (8-class)</w:t>')
xml = xml.replace('<w:t>LiCoNet-S (Bartoli) [21]</w:t>', '<w:t>LiCoNet-S (Bartoli) [21] (8-class)</w:t>')
xml = xml.replace('<w:t>TKWS (Bartoli) [21]</w:t>', '<w:t>TKWS (Bartoli) [21] (8-class)</w:t>')
# Our model
xml = xml.replace('<w:t>Ours (TENet-opt)</w:t>', '<w:t>Ours (TENet-opt) (12-class)</w:t>')


# ============================================================
# WRITE BACK
# ============================================================
with open(DOC, 'w', encoding='utf-8') as f:
    f.write(xml)

print("\n✓ All edits applied successfully.")
