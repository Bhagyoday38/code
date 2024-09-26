import java.util.Random;
import java.util.Scanner;

public class NumberGuessingGame {

    public static void main(String[] args) {
        
        Random random = new Random();
        int randomNumber = random.nextInt(100) + 1;

        
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter your guess (1-100): ");
        int guess = scanner.nextInt();

        while (guess != randomNumber) {
            if (guess < randomNumber) {
                System.out.println("Your guess is too low.");
            } 
            else
                {
                System.out.println("Your guess is too high.");
            }

            
            System.out.println("Enter your guess (1-100): ");
            guess = scanner.nextInt();
        }
        System.out.println("Congratulations! You guessed the correct number.");
    }
}
