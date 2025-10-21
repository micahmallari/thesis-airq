# Video Script: Thesis Progress Report
## "PM2.5 Air Quality Prediction using CapsNet-LSTM-LightGBM Hybrid Models"

---

## **INTRODUCTION (30 seconds)**

**[SLIDE: Title + Team Names]**

"Hi everyone! I'm Andrea, and I'd like to share our thesis progress on developing hybrid models for PM2.5 air quality prediction using satellite images and sensor data. Our team has been working intensively since summer 2024 to implement and improve the methodology from our original proposal."

---

## **ORIGINAL METHODOLOGY vs. CURRENT IMPLEMENTATION (1 minute)**

**[SLIDE: Before vs. After Comparison]**

### **What We Started With (Chapters 1-3):**
- Basic CapsNet for spatial feature extraction
- LSTM for temporal patterns
- Simple LightGBM fusion
- Standard image resizing preprocessing
- Basic cross-validation approach

### **What We've Built:**
- **Enhanced preprocessing pipeline** with both patching AND resizing
- **Modular component architecture** for better collaboration
- **Proper nested cross-validation** with time series splitting
- **Day-specific hyperparameter optimization**
- **Comprehensive data leakage prevention**

---

## **TEAM COLLABORATION & DIVISION OF WORK (45 seconds)**

**[SLIDE: Team Structure Diagram]**

"We strategically divided ourselves into specialized teams:

- **Team 1 (2 people)**: CapsNet/CNN spatial component development
- **Team 2 (2 people)**: LSTM temporal component development

This division allowed us to:
✓ Develop expertise in specific model architectures
✓ Work in parallel on different components
✓ Understand each model's strengths and limitations deeply
✓ Create modular, reusable code components"

---

## **MAJOR PREPROCESSING IMPROVEMENTS (1.5 minutes)**

**[SLIDE: Preprocessing Pipeline Diagram]**

### **1. Enhanced Image Preprocessing**
"Originally, we only planned simple resizing. We implemented:
- **Dual preprocessing approach**: Both patches (256x256) and resized images
- **Advanced augmentation**: Rotation, brightness, contrast adjustments
- **Z-score normalization** for consistent feature scaling
- **Flexible pipeline** supporting different image input strategies"

### **2. Robust Data Matching & Cleaning**
"We built a sophisticated preprocessing pipeline:
- **Timestamp standardization** across all 3 days (7/24, 10/19, 11/10)
- **Spatial-temporal alignment** between images and sensor readings
- **Invalid data detection** and proper handling
- **Multi-location integration** (up to 10 locations per day)"

### **3. Adaptive Feature Handling**
"Our code dynamically adapts to different data availability:
- **7/24 data**: Only PM10 available (temperature/humidity missing)
- **10/19 data**: Complete features but missing some locations
- **11/10 data**: Complete features and all locations
- **Same code, different adaptations** based on available data"

---

## **LSTM COMPONENT DEVELOPMENT (1.5 minutes)**

**[SLIDE: LSTM Architecture Diagram]**

### **Advanced LSTM Implementation**
"Our LSTM component became much more sophisticated:

**Architecture Features:**
- **32-dimensional temporal feature extraction** (optimized for hybrid fusion)
- **Day-specific hyperparameter tuning** (different params for each day)
- **Flexible sequence lengths** (10-60 timesteps based on data patterns)
- **Multiple activation functions** (ReLU, Tanh, GELU)

**Hyperparameter Optimization:**
- **279,936 possible combinations** tested per day
- **Systematic grid search** with validation
- **Example Results**:
  - 7/24: Small network (hidden_size=32, timesteps=30)
  - 10/19: Medium network (hidden_size=64, timesteps=40)  
  - 11/10: Conservative network (high regularization)

**Integration Design:**
- **Modular component** that plugs into main pipeline
- **Feature extraction focus** rather than end-to-end prediction
- **Compatible with different meta-learners** (LightGBM, etc.)"

---

## **PROPER CROSS-VALIDATION METHODOLOGY (1.5 minutes)**

**[SLIDE: Cross-Validation Flow Diagram]**

### **Nested CV Implementation**
"This was a major improvement from our original simple CV approach:

**Data Splitting Strategy:**
- **80% Learning Set**: Used for 5-fold cross-validation
- **20% Holdout Test**: Completely untouched until final evaluation
- **Chronological splitting**: Maintains temporal order

**5-Fold TimeSeriesSplit:**
- **Expanding window approach**: Each fold grows the training set
- **Proper time series validation**: No future data leakage
- **Example split**: Fold 1 (16%→16%), Fold 5 (80%→16%)

**Feature Alignment Solution:**
- **Critical discovery**: LSTM reduces data by timesteps (e.g., -60 samples)
- **Alignment strategy**: All components use same effective indices
- **Verified consistency**: Train/val/test sets properly matched across components"

---

## **ARCHITECTURE INSIGHTS & DISCOVERIES (1 minute)**

**[SLIDE: Model Architecture Diagram]**

### **Component Integration Challenges**
"We learned a lot about hybrid model integration:

**Feature Dimension Consistency:**
- **LSTM**: 32-dimensional temporal features
- **CapsNet**: Configurable spatial features (64-dim planned)
- **Fusion**: Concatenated features → LightGBM meta-learner

**Data Flow Understanding:**
- **Preprocessing** → **Feature Extraction** → **Fusion** → **Prediction**
- **Each component** must handle variable input sizes
- **Pipeline resilience** to missing data scenarios

**Day-Specific Adaptations:**
- **Different optimal architectures** per day based on data characteristics
- **Adaptive input dimensions** (1D for 7/24, 3D for others)
- **Same code, different behaviors** based on available features"

---

## **CURRENT STATUS & NEXT STEPS (1 minute)**

**[SLIDE: Progress Timeline & Next Steps]**

### **✅ Completed:**
- ✓ Complete data preprocessing pipeline
- ✓ LSTM temporal component with hyperparameter tuning
- ✓ Proper nested cross-validation framework
- ✓ Data leakage prevention and validation
- ✓ Modular architecture for team collaboration

### **🔄 In Progress:**
- CapsNet spatial component development (Team 1)
- LightGBM meta-learner optimization
- Feature fusion strategy refinement

### **📅 Next Milestones:**
- **Week 1-2**: Complete CapsNet implementation
- **Week 3**: Full hybrid model integration
- **Week 4**: Comprehensive evaluation across all 3 days
- **Week 5-6**: Results analysis and comparison with baselines

---

## **LESSONS LEARNED & IMPROVEMENTS (45 seconds)**

**[SLIDE: Key Insights]**

### **Technical Insights:**
"Working on this implementation taught us:
- **Importance of proper CV methodology** for time series data
- **Feature alignment challenges** in multi-modal fusion
- **Value of modular architecture** for team collaboration
- **Day-specific optimization** benefits over one-size-fits-all approaches"

### **Collaboration Benefits:**
"Our division of work approach:
- ✓ **Parallel development** increased overall productivity
- ✓ **Specialized expertise** in each component
- ✓ **Better understanding** of model strengths/limitations
- ✓ **Modular code** that's easier to debug and improve"

---

## **CONCLUSION (30 seconds)**

**[SLIDE: Summary & Contact]**

"Our thesis has evolved significantly from the original proposal. We've built a robust, modular hybrid modeling pipeline with proper validation methodology. The summer development phase gave us deep insights into both the technical challenges and the collaborative aspects of machine learning research.

We're excited to complete the remaining components and see how our enhanced methodology performs compared to the original baseline approaches. Thank you for your attention!"

---

## **TECHNICAL APPENDIX (If Questions)**

### **Code Structure:**
```
thesis-airq/
├── preprocessing/          # Data cleaning & matching
├── i_components/lstm/     # Modular LSTM component  
├── hybrid_models/         # Full pipeline integration
├── dataset/              # Multi-stage processed data
└── models/lstm_temporal/ # Day-specific parameters
```

### **Key Innovations:**
1. **Dual preprocessing**: Patches + resize
2. **Day-adaptive parameters**: Optimal LSTM per day
3. **Proper time series CV**: 80-20 with 5-fold on learning
4. **Feature alignment**: Consistent indices across components
5. **Data leakage prevention**: Comprehensive validation

---

**Video Length: ~8-9 minutes**
**Format: Technical presentation with slides + code demos**
**Audience: Academic advisors + technical peers**
