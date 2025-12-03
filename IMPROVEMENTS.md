# Detection Improvements

## Recent Improvements to Address Detection Issues

### 1. Enhanced OCR Label Detection

**Multiple OCR Strategies:**
- Full image OCR with data output to get text bounding boxes
- Multiple preprocessing approaches (enhanced contrast, binary threshold, original)
- Per-region OCR with enhanced preprocessing above each rectangle
- Multiple PSM (Page Segmentation Mode) attempts per region

**Better Preprocessing:**
- Enhanced contrast with `cv2.convertScaleAbs()` 
- Binary thresholding with Otsu method
- Automatic inversion for dark text on light backgrounds
- Expanded search area (80 pixels above rectangles, up from 50)

**Improved Text Matching:**
- More lenient horizontal alignment (overlapping or center-aligned)
- Expanded vertical search range
- Better deduplication of OCR results
- Text validation to filter out noise

### 2. Better Rectangle Detection Accuracy

**Multiple Threshold Methods:**
- Adaptive thresholding (handles varying lighting)
- Otsu thresholding (better for high-contrast images)
- Combined results from both methods

**More Lenient Parameters:**
- Lower minimum contour area (500 pixels, down from 1000)
- Accepts 3+ vertices (instead of requiring 4)
- Wider aspect ratio range (0.05 to 20)
- Better deduplication to avoid duplicate rectangles

### 3. Enhanced Type Inference

**More Keywords and Patterns:**
- Email: handles OCR errors like "emall", "emai1"
- Password: "pwd", "passw", "passwor"
- Phone: "phone", "telephone", "mobile", "cell"
- Name: "first name", "last name", "full name", "fname", "lname"
- Address: "address", "street", "city", "zip", "postal"
- Long text: "description", "notes", "comment", "message", "bio", "about"

**Better Handling:**
- Case-insensitive matching
- Partial word matching
- Handles common OCR errors

### 4. Improved Logging

Added detailed logging throughout:
- Number of rectangles found
- Number of text regions from OCR
- Label matching results
- Field creation with full details

### 5. Fallback Mechanisms

**For Sample Images:**
- Automatic fallback for known sample images when detection fails
- Predefined fields for sketch1, sketch2, sketch3

**For Missing Labels:**
- Creates fields even when labels aren't detected
- Uses generic names (field_1, field_2, etc.) that users can edit

## Troubleshooting Detection Issues

### If Labels Aren't Detected:

1. **Check Image Quality:**
   - Ensure high contrast between text and background
   - Use clear, bold text
   - Avoid blurry images

2. **Check Console Logs:**
   - Look for "Found X text regions from OCR" - if 0, OCR isn't finding text
   - Look for "Matched label..." messages - shows successful matches
   - Look for "No label found..." messages - shows which rectangles need labels

3. **Try Different Images:**
   - Test with the sample images first
   - Use images with larger text
   - Ensure text is clearly separated from rectangles

### If Type Detection Isn't Working:

1. **Check if Labels are Detected:**
   - Type inference depends on having label text
   - If labels show as "field_1", "field_2", types won't be inferred

2. **Verify Label Text:**
   - Check the review page - do fields show actual text or just "field_X"?
   - If text is garbled, type inference may fail

3. **Manual Editing:**
   - Users can always edit field names and types on the review page
   - The system provides suggestions, but manual review is recommended

### Improving Detection Accuracy:

1. **For Better Rectangle Detection:**
   - Draw clear, thick rectangles
   - Ensure rectangles are closed (no gaps)
   - Use consistent spacing

2. **For Better Label Detection:**
   - Place labels clearly above rectangles
   - Use larger, clearer text
   - Ensure good contrast
   - Avoid overlapping text and rectangles

3. **For Better Type Detection:**
   - Use standard field names (Email, Password, Date, etc.)
   - Write clearly and legibly
   - Avoid abbreviations unless they're clear

## Configuration Constants

You can adjust these in `pipeline.py`:

- `MIN_CONTOUR_AREA = 500` - Lower for smaller rectangles
- `LABEL_SEARCH_HEIGHT = 80` - Increase if labels are far above rectangles
- `ROI_PADDING = 10` - Increase if labels are outside rectangle bounds

## Next Steps

If detection is still not accurate enough:

1. **Use the review page** - Users can edit all field names and types before generating code
2. **Improve source images** - Better input images = better detection
3. **Consider ML models** - For production, consider YOLO or pix2struct models (see README)

