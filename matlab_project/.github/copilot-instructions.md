# GitHub Copilot Instructions for MATLAB Deep Learning Project

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Context
This is a MATLAB deep learning project implementing a Generative Adversarial Network (GAN) for PZT (Piezoelectric) sensor data analysis. The primary challenge is maintaining data integrity during batch processing to prevent splitting of time series cycles.

## Critical Requirements

### Data Handling - ABSOLUTELY CRITICAL
- **Each data sample MUST maintain structure: 4000×2×6 (time_steps × sensors × channels)**
- **NEVER split the 4000 time steps** - this is the core issue we're solving
- **Use "SSCB" format consistently** throughout the entire pipeline
- **Keep MiniBatchSize small (1-4)** to prevent memory issues and maintain data integrity
- **Every data transformation must preserve the complete cycle structure**

### Network Architecture
- GAN with attention mechanisms for time series data
- Handle matrix dimension mismatches carefully (transpose operations when needed)
- Include comprehensive debug output for monitoring tensor shapes
- Design layers to work with 4000×2×6 input dimensions

### MATLAB-Specific Best Practices
- Use .m file extensions for all MATLAB scripts
- Follow MATLAB naming conventions (camelCase for functions)
- Include proper error handling with try-catch blocks
- Add size() and class() checks at critical data flow points
- Use fprintf() for tracking tensor dimensions during training

### Common Pitfalls to Avoid
1. **Automatic dimension reordering**: MATLAB may change dimensions - ensure consistency
2. **Batch splitting**: MiniBatch affects grouping, NOT splitting of time steps
3. **Memory allocation**: Large matrices require careful memory management
4. **trainingOptions format mismatch**: Always specify 'OutputFormat', 'SSCB'

### Debugging Requirements
- Add debugDataFlow() calls at every major data transformation
- Include timing measurements with tic/toc or custom timing utilities
- Log batch shapes, data ranges, and tensor dimensions
- Verify data integrity after each processing step

### File Organization Standards
- `src/data/`: Data preprocessing and custom datastore creation
- `src/models/`: Network architectures (Generator, Discriminator, etc.)
- `src/training/`: Training scripts and configuration
- `src/utils/`: Helper functions, debugging, and utilities
- `data/`: Raw and processed data files
- `results/`: Training outputs, models, and logs

### Code Assistance Guidelines
When suggesting code changes:
1. **Always consider impact on data flow and tensor shapes**
2. **Provide debug output to verify changes work correctly**
3. **Test with small batch sizes first**
4. **Maintain compatibility with MATLAB Deep Learning Toolbox**
5. **Include comprehensive error handling**

### Example Correct Data Flow
```matlab
% Correct data flow - maintain cycle integrity
raw_cycle: [4000, 2, 6] per cycle
multiple_cycles: [4000, 2, 6, num_cycles]  
batched_data: [4000, 2, 6, batch_size] % NEVER [1, 2, 6, 4000*batch_size]
network_input: same dimensions preserved through all layers
```

### Priority Areas for Assistance
1. **Data batching and shape preservation** (highest priority)
2. **Custom datastore implementation for cycle integrity**
3. **GAN architecture debugging and optimization**
4. **Training loop optimization and monitoring**
5. **Memory management for large time series tensors**
6. **Performance profiling and bottleneck identification**

### When in Doubt
- **Always preserve the 4000×2×6 structure**
- **Add debug output to verify tensor shapes**
- **Use SSCB format in all training configurations**
- **Test with MiniBatchSize=1 first**
- **Include comprehensive error messages**
