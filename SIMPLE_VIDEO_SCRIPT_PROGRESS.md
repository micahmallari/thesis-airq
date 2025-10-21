# Simple Video Script: Thesis Progress Report
## "PM2.5 Air Quality Prediction using Hybrid Models - Our Journey So Far"

---

## **INTRODUCTION (30 seconds)**

**[SLIDE: Title + Team Photo]**

"Hi! I'm Andrea, and I want to share our thesis progress over the past few months. We're developing hybrid models for PM2.5 air quality prediction using satellite images and sensor data. Let me walk you through what we've accomplished since summer."

---

## **TEAM STRATEGY & SUMMER DEVELOPMENT (1 minute)**

**[SLIDE: Team Division Diagram]**

### **Our Collaborative Approach:**
"We strategically divided our team into two specialized groups:

- **Team 1 (2 people)**: CapsNet & CNN spatial components
- **Team 2 (2 people)**: LSTM temporal components

**Why this worked:**
✓ **Parallel development** - We could work simultaneously on different parts
✓ **Deep understanding** - Each team became experts in their specific models
✓ **Better learning** - We really understood how each component works
✓ **Faster progress** - No waiting for others to finish before starting"

### **Summer Coding Experience:**
"Starting in summer helped us realize the complexity. As we coded, we understood the data flow, cross-validation challenges, and how components need to work together."

---

## **PREPROCESSING IMPROVEMENTS (1.5 minutes)**

**[SLIDE: Before vs After Preprocessing]**

### **Research-Driven Changes:**

**Original Plan (from Chapters 1-3):**
- Simple image resizing
- Basic data cleaning

**What We Actually Implemented:**
- **Added Patching**: We found research showing patching works better for high-resolution satellite images
- **Dual Approach**: Both patches (256x256) AND resizing for comparison
- **Data Augmentation**: To increase our dataset since we have limited images

### **Why Patching?**
"When we researched preprocessing for satellite images, we discovered:
- **High-resolution benefits**: Patches preserve local spatial patterns
- **Data multiplication**: One image becomes multiple patches
- **Better for small datasets**: Increases training data significantly

So we implemented both patching and resizing to compare which works better."

### **Advanced Data Pipeline:**
"We built a complete preprocessing system:
- **Timestamp matching** between images and sensor data
- **Multi-location handling** (up to 10 locations per day)
- **Data quality checks** and invalid data filtering
- **Chronological splitting** for proper time series handling"

---

## **UNDERSTANDING DATA STRUCTURE (1 minute)**

**[SLIDE: Data Characteristics by Day]**

### **Real Data Challenges:**
"As we implemented, we discovered our 3 days have different characteristics:

- **7/24 data**: Only PM10 available (no temperature/humidity), 5 locations
- **10/19 data**: Complete features, 9 locations  
- **11/10 data**: Complete features, all 10 locations

**Solution**: Our code adapts automatically!
- Same LSTM code handles 1D or 3D input
- Different optimal parameters for each day
- Robust to missing data scenarios"

---

## **CROSS-VALIDATION METHODOLOGY (1.5 minutes)**

**[SLIDE: CV Diagram - Before vs After]**

### **Major Improvement from Original Thesis:**

**Original Approach**: Simple train-test split
**Current Approach**: Proper nested cross-validation

### **What We Learned:**
"Time series data needs special handling:
- **80% Learning Set**: For 5-fold cross-validation
- **20% Holdout Test**: Never touched until final evaluation
- **Expanding Window**: Each CV fold gets more historical data
- **No Data Leakage**: Future data never influences past predictions

### **Real Implementation:**
- **Fold 1**: Train on 10k samples → Validate on next 10k
- **Fold 4**: Train on 41k samples → Validate on next 10k  
- **Growing training sets** = Better temporal pattern learning"

---

## **COMPONENT DEVELOPMENT INSIGHTS (1.5 minutes)**

**[SLIDE: Component Architecture]**

### **LSTM Team Discoveries:**
"Our LSTM team built a sophisticated component:
- **32-dimensional feature extraction** (optimized for fusion)
- **Day-specific hyperparameters** (each day gets optimal settings)
- **279,936 parameter combinations** tested per day
- **Modular design** that plugs into the main pipeline

**Example Results:**
- 7/24: Small network (hidden_size=32) for limited features
- 10/19: Medium network (hidden_size=64) for complete data
- 11/10: Conservative network with high regularization"

### **CapsNet/CNN Team Progress:**
"Our spatial team implemented:
- **CapsNet architecture** for spatial feature extraction
- **CNN baseline** for comparison (ResNet50, EfficientNet)
- **Flexible feature dimensions** to match LSTM output
- **Both patch and resize processing** pipelines"

---

## **PROJECT TRACKING & ORGANIZATION (45 seconds)**

**[SLIDE: Project Structure]**

### **Staying Organized:**
"We created systematic tracking:
- **Modular code structure** (`i_components/`, `preprocessing/`, `hybrid_models/`)
- **Documentation files** for methodology corrections
- **Version control** for team collaboration
- **Regular progress reviews** to align components

**Benefits:**
✓ **Clear responsibilities** - Each team knows their scope
✓ **Easy integration** - Components designed to work together
✓ **Progress visibility** - Everyone sees overall advancement
✓ **Quality control** - Systematic testing and validation"

---

## **CURRENT STATUS & NEXT STEPS (1 minute)**

**[SLIDE: Progress Timeline]**

### **✅ Completed:**
- ✓ Complete data preprocessing pipeline with patching
- ✓ LSTM component with day-specific optimization  
- ✓ Proper nested cross-validation framework
- ✓ CapsNet and CNN spatial components
- ✓ Data quality validation and leakage prevention

### **🔄 In Progress:**
- Final hybrid model integration
- Feature fusion optimization  
- Comprehensive evaluation across all days

### **📅 Next 2-3 Weeks:**
- **Week 1**: Complete model integration testing
- **Week 2**: Run full evaluation pipeline
- **Week 3**: Results analysis and comparison with baselines

---

## **LESSONS LEARNED (45 seconds)**

**[SLIDE: Key Insights]**

### **Technical Lessons:**
"What we learned that wasn't in our original chapters:
- **Preprocessing matters more** than we expected
- **Time series cross-validation** is completely different from regular CV
- **Feature alignment** between components is critical
- **Day-specific optimization** beats one-size-fits-all approaches"

### **Collaboration Lessons:**
"Team division strategy worked great:
- ✓ **Faster development** through parallel work
- ✓ **Deeper expertise** in each component
- ✓ **Better problem-solving** when we combine knowledge
- ✓ **Shared learning** about hybrid model challenges"

---

## **CONCLUSION (30 seconds)**

**[SLIDE: Summary]**

"Our thesis has evolved significantly from the original proposal. We've built a robust, research-driven preprocessing pipeline, implemented proper time series validation, and developed modular components that work together seamlessly.

The summer development phase was crucial - it helped us understand the real challenges and implement solutions that go beyond what we initially planned. We're excited to see the final results!"

**Thank you!**

---

## **QUICK Q&A PREP**

**Potential Questions:**
- **"How different is this from your original methodology?"** → Show preprocessing evolution and CV improvements
- **"What was the biggest challenge?"** → Feature alignment and time series CV
- **"How did team division help?"** → Parallel development and specialized expertise
- **"What's your expected timeline?"** → 2-3 weeks for complete results

---

**Video Length: ~7-8 minutes**  
**Format: Simple slides + brief code demos**  
**Tone: Conversational, showing real progress and learning**
