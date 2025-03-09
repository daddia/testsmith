/**
 * Complex JavaScript module with more sophisticated functions.
 */

/**
 * Custom error for data validation
 */
class DataValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = 'DataValidationError';
  }
}

/**
 * Calculate the average of an array of numbers
 * @param {number[]} numbers - Array of numbers
 * @returns {number} Average value
 * @throws {Error} If the array is empty
 */
function calculateAverage(numbers) {
  if (!numbers || numbers.length === 0) {
    throw new Error("Cannot calculate average of empty array");
  }
  
  const sum = numbers.reduce((acc, val) => acc + val, 0);
  return sum / numbers.length;
}

/**
 * Validate an email address using regex
 * @param {string} email - Email address to validate
 * @returns {boolean} True if valid, false otherwise
 */
function validateEmail(email) {
  const pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  return pattern.test(email);
}

/**
 * Process user data with validation and transformation
 * @param {Object} data - User data object
 * @param {string} data.name - User's name
 * @param {string} data.email - User's email
 * @param {number} data.age - User's age
 * @param {string[]} [data.interests] - User's interests (optional)
 * @returns {Object} Processed data
 * @throws {DataValidationError} If required fields are missing or invalid
 */
function processUserData(data) {
  // Validate required fields
  const requiredFields = ['name', 'email', 'age'];
  for (const field of requiredFields) {
    if (!(field in data)) {
      throw new DataValidationError(`Missing required field: ${field}`);
    }
  }
  
  // Validate email
  if (!validateEmail(data.email)) {
    throw new DataValidationError(`Invalid email: ${data.email}`);
  }
  
  // Validate age
  if (typeof data.age !== 'number' || data.age < 0) {
    throw new DataValidationError(`Invalid age: ${data.age}`);
  }
  
  // Process data
  const result = {
    name: data.name.trim().split(' ').map(part => 
      part.charAt(0).toUpperCase() + part.slice(1).toLowerCase()
    ).join(' '),
    email: data.email.toLowerCase(),
    age: data.age,
    isAdult: data.age >= 18,
    processedAt: new Date().toISOString(),
  };
  
  // Add optional fields
  if (data.interests && Array.isArray(data.interests)) {
    result.interests = data.interests.map(interest => interest.toLowerCase());
  }
  
  // Generate user ID
  result.userId = hashString(`${result.name}:${result.email}`);
  
  console.log(`Processed user data for ${result.name} (ID: ${result.userId})`);
  return result;
}

/**
 * Simple string hashing function (for demo purposes)
 * @param {string} str - String to hash
 * @returns {string} Hashed string
 */
function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash).toString(16);
}

/**
 * Analyze the sentiment of a text (mock implementation)
 * @param {string} text - Text to analyze
 * @returns {[string, number]} Tuple of [sentiment, confidence]
 */
function analyzeTextSentiment(text) {
  const positiveWords = ['good', 'great', 'excellent', 'awesome', 'happy', 'love', 'best'];
  const negativeWords = ['bad', 'terrible', 'awful', 'sad', 'hate', 'worst', 'poor'];
  
  const textLower = text.toLowerCase();
  const words = textLower.match(/\w+/g) || [];
  
  const positiveCount = words.filter(word => positiveWords.includes(word)).length;
  const negativeCount = words.filter(word => negativeWords.includes(word)).length;
  
  const totalSentimentWords = positiveCount + negativeCount;
  if (totalSentimentWords === 0) {
    return ['neutral', 0.5];
  }
  
  if (positiveCount > negativeCount) {
    const confidence = positiveCount / totalSentimentWords;
    return ['positive', confidence];
  } else if (negativeCount > positiveCount) {
    const confidence = negativeCount / totalSentimentWords;
    return ['negative', confidence];
  } else {
    return ['neutral', 0.5];
  }
}

/**
 * Create a memoized version of a function with expiration
 * @param {Function} fn - Function to memoize
 * @param {number} [expirySeconds=60] - Time in seconds before cache expires
 * @returns {Function} Memoized function
 */
function memoize(fn, expirySeconds = 60) {
  const cache = new Map();
  
  return function(...args) {
    const key = JSON.stringify(args);
    
    if (cache.has(key)) {
      const [result, timestamp] = cache.get(key);
      if ((Date.now() - timestamp) / 1000 < expirySeconds) {
        console.log(`Cache hit for ${fn.name}`);
        return result;
      }
    }
    
    const result = fn.apply(this, args);
    cache.set(key, [result, Date.now()]);
    console.log(`Cache miss for ${fn.name}, storing result`);
    return result;
  };
}

/**
 * Perform an expensive computation (simulated)
 * @param {number} n - Input parameter
 * @returns {number} Result of computation
 */
function expensiveComputation(n) {
  console.log(`Performing expensive computation with n=${n}`);
  
  // Simulate expensive computation
  let result = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      result += i * j;
    }
  }
  
  return result;
}

// Create memoized version of the expensive computation
const memoizedComputation = memoize(expensiveComputation, 300);

module.exports = {
  DataValidationError,
  calculateAverage,
  validateEmail,
  processUserData,
  analyzeTextSentiment,
  memoize,
  expensiveComputation,
  memoizedComputation
};