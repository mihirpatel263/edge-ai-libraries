# Release Notes


## Current Release

**Version**: 1.2.0 \
**Release Date**: WW20 2025 

- Support for GPU (discrete and integrated) is now available. Refer to system requirements documentation for details.
- Bug fixes

### Known Issues/Behavior:
- DeepSeek/Phi Models are observed, at times, to continue generating response in an endless loop. Close the browser and restart in such cases.

## Previous releases

**Version**: 1.1.2 \
**Release Date**: WW16 2025 

- Edge Orchestrator onboarding supported. Documentation updated to provide necessary onboarding process details.
- Persistent volume used instead of hostpath. This is enabled by default requiring clusters to support dynamic storage support.
- Documentation updated for ESC compatability. As ESC supports only absolute file path, the links in the documentation will always point to main repo even on forked repos.
- Bug fixes

## Previous releases

**Version**: 1.1.1 \
**Release Date**: WW13 2025 

- Updated the documentation to reflect availability in public artefactory.
- Bug fixes.

**Version**: 1.0.0 \
**Release Date**: WW11 2025 

- Initial release of the ChatQ&A Sample Application.
- Added support for vLLM, TGI, and OVMS inference methods.
- Improved user interface for better user experience.