# PatientX.AI - Expected Data Format

This tool supports input data in two formats: **CSV Directory** and **Text File**. Please ensure your data follows the required structure for accurate processing.

---

## 📁 CSV Directory Input

When using a directory containing CSV files, each CSV must adhere to the following structure:

### Required Columns:
- **forum**: Name of the forum from which the data is collected.
- **thread_title**: Title of the discussion thread.
- **message_nr**: Message number within the thread.
- **post_message**: Text content of the message.

### Example CSV Structure:
```csv
forum,thread_title,message_nr,post_message
ForumA,Introduction,1,Hello everyone! Excited to join this community.
ForumA,Introduction,2,Welcome! Glad to have you here.
ForumB,General Discussion,1,What's your experience with topic modeling?
ForumB,General Discussion,2,I found BERTopic to work well with messy data.
```
**Directory Structure:** Ensure all CSV files are placed in a single directory. Every file must follow the column structure above.

---

## 📄 Text File Input

Alternatively, a plain text file can be used, where **each new line** is considered a distinct document.

### Example Text File:
```txt
Hello everyone! Excited to join this community.
Welcome! Glad to have you here.
What's your experience with topic modeling?
I found BERTopic to work well with messy data.
```

- File Structure: Ensure each line is a standalone message or document. Avoid using line breaks within a single document.

---

## 🔎 Summary of Input Requirements

| Input Type    | File Format | Structure Requirements                                                                                          |
|---------------|-------------|-----------------------------------------------------------------------------------------------------------------|
| CSV Directory | `.csv`      | Must include **forum**, **thread_title**, **message_nr**, and **post_message** columns. All files in one directory. |
| Text File     | `.txt`      | Each line represents a distinct document. Avoid multi-line entries for a single document.                       |
