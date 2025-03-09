<?php
/**
 * Complex PHP module with more sophisticated functions.
 */

/**
 * Custom exception for data validation errors
 */
class DataValidationException extends \Exception {
    public function __construct($message, $code = 0, \Exception $previous = null) {
        parent::__construct($message, $code, $previous);
    }
}

/**
 * Calculate the average of an array of numbers
 *
 * @param array $numbers Array of numbers
 * @return float Average value
 * @throws \Exception If the array is empty
 */
function calculateAverage(array $numbers) {
    if (empty($numbers)) {
        throw new \Exception("Cannot calculate average of empty array");
    }
    
    return array_sum($numbers) / count($numbers);
}

/**
 * Validate an email address using regex
 *
 * @param string $email Email address to validate
 * @return bool True if valid, false otherwise
 */
function validateEmail($email) {
    $pattern = '/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/';
    return (bool) preg_match($pattern, $email);
}

/**
 * Process user data with validation and transformation
 *
 * @param array $data User data array
 * @return array Processed data
 * @throws DataValidationException If required fields are missing or invalid
 */
function processUserData(array $data) {
    // Validate required fields
    $requiredFields = ['name', 'email', 'age'];
    foreach ($requiredFields as $field) {
        if (!isset($data[$field])) {
            throw new DataValidationException("Missing required field: {$field}");
        }
    }
    
    // Validate email
    if (!validateEmail($data['email'])) {
        throw new DataValidationException("Invalid email: {$data['email']}");
    }
    
    // Validate age
    if (!is_numeric($data['age']) || $data['age'] < 0) {
        throw new DataValidationException("Invalid age: {$data['age']}");
    }
    
    // Process data
    $result = [
        'name' => ucwords(trim($data['name'])),
        'email' => strtolower($data['email']),
        'age' => (int) $data['age'],
        'is_adult' => $data['age'] >= 18,
        'processed_at' => date('c'),
    ];
    
    // Add optional fields
    if (isset($data['interests']) && is_array($data['interests'])) {
        $result['interests'] = array_map('strtolower', $data['interests']);
    }
    
    // Generate user ID
    $result['user_id'] = md5("{$result['name']}:{$result['email']}");
    
    error_log("Processed user data for {$result['name']} (ID: {$result['user_id']})");
    return $result;
}

/**
 * Analyze the sentiment of a text (mock implementation)
 *
 * @param string $text Text to analyze
 * @return array Array containing [sentiment, confidence]
 */
function analyzeTextSentiment($text) {
    $positiveWords = ['good', 'great', 'excellent', 'awesome', 'happy', 'love', 'best'];
    $negativeWords = ['bad', 'terrible', 'awful', 'sad', 'hate', 'worst', 'poor'];
    
    $textLower = strtolower($text);
    preg_match_all('/\w+/', $textLower, $matches);
    $words = $matches[0];
    
    $positiveCount = 0;
    $negativeCount = 0;
    
    foreach ($words as $word) {
        if (in_array($word, $positiveWords)) {
            $positiveCount++;
        } elseif (in_array($word, $negativeWords)) {
            $negativeCount++;
        }
    }
    
    $totalSentimentWords = $positiveCount + $negativeCount;
    if ($totalSentimentWords === 0) {
        return ['neutral', 0.5];
    }
    
    if ($positiveCount > $negativeCount) {
        $confidence = $positiveCount / $totalSentimentWords;
        return ['positive', $confidence];
    } elseif ($negativeCount > $positiveCount) {
        $confidence = $negativeCount / $totalSentimentWords;
        return ['negative', $confidence];
    } else {
        return ['neutral', 0.5];
    }
}

/**
 * Cache implementation with expiration
 */
class Cache {
    private $data = [];
    
    /**
     * Store a value in the cache
     *
     * @param string $key Cache key
     * @param mixed $value Value to store
     * @param int $expirySeconds Seconds until expiration
     */
    public function set($key, $value, $expirySeconds = 60) {
        $this->data[$key] = [
            'value' => $value,
            'expires' => time() + $expirySeconds
        ];
    }
    
    /**
     * Retrieve a value from the cache
     *
     * @param string $key Cache key
     * @return mixed|null Value or null if not found or expired
     */
    public function get($key) {
        if (isset($this->data[$key])) {
            $item = $this->data[$key];
            if (time() < $item['expires']) {
                return $item['value'];
            }
            // Expired, remove it
            unset($this->data[$key]);
        }
        return null;
    }
    
    /**
     * Check if a key exists and is not expired
     *
     * @param string $key Cache key
     * @return bool True if key exists and is not expired
     */
    public function has($key) {
        if (isset($this->data[$key])) {
            return time() < $this->data[$key]['expires'];
        }
        return false;
    }
    
    /**
     * Remove a key from the cache
     *
     * @param string $key Cache key
     */
    public function delete($key) {
        unset($this->data[$key]);
    }
    
    /**
     * Clear all expired items
     */
    public function cleanup() {
        $now = time();
        foreach ($this->data as $key => $item) {
            if ($now >= $item['expires']) {
                unset($this->data[$key]);
            }
        }
    }
}

/**
 * Perform an expensive computation (simulated)
 *
 * @param int $n Input parameter
 * @return int Result of computation
 */
function expensiveComputation($n) {
    error_log("Performing expensive computation with n={$n}");
    
    // Simulate expensive computation
    $result = 0;
    for ($i = 0; $i < $n; $i++) {
        for ($j = 0; $j < $n; $j++) {
            $result += $i * $j;
        }
    }
    
    return $result;
}

/**
 * Cache implementation for the expensive computation
 */
$computationCache = new Cache();

/**
 * Cached version of the expensive computation
 *
 * @param int $n Input parameter
 * @return int Result of computation
 */
function cachedComputation($n) {
    global $computationCache;
    
    $cacheKey = "computation_{$n}";
    
    if ($computationCache->has($cacheKey)) {
        error_log("Cache hit for expensiveComputation");
        return $computationCache->get($cacheKey);
    }
    
    $result = expensiveComputation($n);
    $computationCache->set($cacheKey, $result, 300);
    error_log("Cache miss for expensiveComputation, storing result");
    
    return $result;
}
?>