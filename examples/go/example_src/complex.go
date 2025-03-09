package complex

import (
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"
)

// DataValidationError represents an error that occurs during data validation.
type DataValidationError struct {
	Field   string
	Message string
}

// Error returns the string representation of the error.
func (e *DataValidationError) Error() string {
	return fmt.Sprintf("Validation error for field '%s': %s", e.Field, e.Message)
}

// NewDataValidationError creates a new DataValidationError.
func NewDataValidationError(field, message string) *DataValidationError {
	return &DataValidationError{
		Field:   field,
		Message: message,
	}
}

// CalculateAverage calculates the average of a slice of numbers.
// Returns an error if the slice is empty.
func CalculateAverage(numbers []float64) (float64, error) {
	if len(numbers) == 0 {
		return 0, errors.New("cannot calculate average of empty slice")
	}

	sum := 0.0
	for _, num := range numbers {
		sum += num
	}

	return sum / float64(len(numbers)), nil
}

// ValidateEmail validates an email address using regex.
func ValidateEmail(email string) bool {
	pattern := `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
	regex := regexp.MustCompile(pattern)
	return regex.MatchString(email)
}

// UserData represents user information.
type UserData struct {
	Name      string   `json:"name"`
	Email     string   `json:"email"`
	Age       int      `json:"age"`
	IsAdult   bool     `json:"is_adult,omitempty"`
	Interests []string `json:"interests,omitempty"`
	UserID    string   `json:"user_id,omitempty"`
	ProcessedAt string `json:"processed_at,omitempty"`
}

// ProcessUserData processes user data with validation and transformation.
func ProcessUserData(data UserData) (UserData, error) {
	// Validate required fields
	if data.Name == "" {
		return UserData{}, NewDataValidationError("name", "Name is required")
	}
	
	if data.Email == "" {
		return UserData{}, NewDataValidationError("email", "Email is required")
	}
	
	if !ValidateEmail(data.Email) {
		return UserData{}, NewDataValidationError("email", fmt.Sprintf("Invalid email: %s", data.Email))
	}
	
	if data.Age < 0 {
		return UserData{}, NewDataValidationError("age", fmt.Sprintf("Invalid age: %d", data.Age))
	}
	
	// Process data
	result := UserData{
		Name:        strings.TrimSpace(data.Name),
		Email:       strings.ToLower(data.Email),
		Age:         data.Age,
		IsAdult:     data.Age >= 18,
		ProcessedAt: time.Now().Format(time.RFC3339),
	}
	
	// Capitalize first letter of each word in name
	nameParts := strings.Fields(result.Name)
	for i, part := range nameParts {
		if len(part) > 0 {
			nameParts[i] = strings.ToUpper(part[:1]) + strings.ToLower(part[1:])
		}
	}
	result.Name = strings.Join(nameParts, " ")
	
	// Add optional fields
	if len(data.Interests) > 0 {
		result.Interests = make([]string, len(data.Interests))
		for i, interest := range data.Interests {
			result.Interests[i] = strings.ToLower(interest)
		}
	}
	
	// Generate user ID (simplified for example)
	result.UserID = fmt.Sprintf("%x", strings.NewReader(result.Name+":"+result.Email).Size())
	
	fmt.Printf("Processed user data for %s (ID: %s)\n", result.Name, result.UserID)
	return result, nil
}

// SentimentAnalysis represents the result of sentiment analysis.
type SentimentAnalysis struct {
	Sentiment  string  `json:"sentiment"`
	Confidence float64 `json:"confidence"`
}

// AnalyzeTextSentiment analyzes the sentiment of a text (mock implementation).
func AnalyzeTextSentiment(text string) SentimentAnalysis {
	positiveWords := []string{"good", "great", "excellent", "awesome", "happy", "love", "best"}
	negativeWords := []string{"bad", "terrible", "awful", "sad", "hate", "worst", "poor"}
	
	textLower := strings.ToLower(text)
	words := regexp.MustCompile(`\w+`).FindAllString(textLower, -1)
	
	positiveCount := 0
	negativeCount := 0
	
	for _, word := range words {
		for _, positive := range positiveWords {
			if word == positive {
				positiveCount++
				break
			}
		}
		
		for _, negative := range negativeWords {
			if word == negative {
				negativeCount++
				break
			}
		}
	}
	
	totalSentimentWords := positiveCount + negativeCount
	if totalSentimentWords == 0 {
		return SentimentAnalysis{
			Sentiment:  "neutral",
			Confidence: 0.5,
		}
	}
	
	if positiveCount > negativeCount {
		return SentimentAnalysis{
			Sentiment:  "positive",
			Confidence: float64(positiveCount) / float64(totalSentimentWords),
		}
	} else if negativeCount > positiveCount {
		return SentimentAnalysis{
			Sentiment:  "negative",
			Confidence: float64(negativeCount) / float64(totalSentimentWords),
		}
	} else {
		return SentimentAnalysis{
			Sentiment:  "neutral",
			Confidence: 0.5,
		}
	}
}

// CacheItem represents an item in the cache.
type CacheItem struct {
	Value     interface{}
	ExpiresAt time.Time
}

// Cache represents a simple in-memory cache with expiration.
type Cache struct {
	data map[string]CacheItem
}

// NewCache creates a new cache.
func NewCache() *Cache {
	return &Cache{
		data: make(map[string]CacheItem),
	}
}

// Set stores a value in the cache with an expiration time.
func (c *Cache) Set(key string, value interface{}, expirySeconds int) {
	c.data[key] = CacheItem{
		Value:     value,
		ExpiresAt: time.Now().Add(time.Duration(expirySeconds) * time.Second),
	}
}

// Get retrieves a value from the cache.
func (c *Cache) Get(key string) (interface{}, bool) {
	item, exists := c.data[key]
	if !exists {
		return nil, false
	}
	
	if time.Now().After(item.ExpiresAt) {
		delete(c.data, key)
		return nil, false
	}
	
	return item.Value, true
}

// ExpensiveComputation performs an expensive computation (simulated).
func ExpensiveComputation(n int) int {
	fmt.Printf("Performing expensive computation with n=%d\n", n)
	
	// Simulate expensive computation
	result := 0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			result += i * j
		}
	}
	
	return result
}