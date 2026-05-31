# SPDX-License-Identifier: Apache-2.0
"""napari-based GUI layer. Calls into ``proteia.core``; never the reverse.

napari/Qt imports are kept lazy (inside functions) so that importing this
package does not require a display.
"""
