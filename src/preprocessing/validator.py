def validate_comment(comment: dict) -> tuple[bool, str]:
    if not isinstance(comment, dict):
        return False, "Comment is not a dictionary"
    
    text = comment.get("text")
    if not isinstance(text, str):
        return False, "Text field is missing or not a string"
        
    if not text.strip():
        return False, "Text field is empty or whitespace only"
        
    return True, "Valid"

def validate_comments_batch(comments: list) -> tuple[list, list]:
    valid = []
    invalid = []
    
    if not isinstance(comments, list):
        return valid, invalid
        
    for comment in comments:
        is_valid, _ = validate_comment(comment)
        if is_valid:
            valid.append(comment)
        else:
            invalid.append(comment)
            
    return valid, invalid
