import java.util.Random;
import java.util.Scanner;

public class Hangman {

    public static void main(String[] args) {
        
        String[] words = {"apple", "banana", "cherry", "dog", "elephant", "fish", "grape", "horse", "ice cream", "juice", "kiwi", "lemon", "mango", "nut", "orange", "peach", "pear", "pineapple", "queen", "rabbit", "strawberry", "tiger", "umbrella", "violin", "watermelon", "x-ray", "yacht", "zebra"};

        
        Random random = new Random();
        String word = words[random.nextInt(words.length)];

       
        String guesses = "";

        
        int guessesLeft = 6;

        
        while (guessesLeft > 0) {
            
            System.out.println("Word: " + getGuessedWord(word, guesses));

            
            System.out.println("Guess a letter:");
            Scanner scanner = new Scanner(System.in);
            String guess = scanner.nextLine();

           
            if (word.contains(guess)) {
                
                guesses += guess;

                
                if (getGuessedWord(word, guesses).equals(word)) {
                    System.out.println("You win!");
                    break;
                }
            } else {
               
                guessesLeft--;
                System.out.println("Incorrect guess. You have " + guessesLeft + " guesses left.");
            }
        }

        
        if (guessesLeft == 0) {
            System.out.println("You lose!");
        }
    }
