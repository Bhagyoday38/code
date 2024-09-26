import java.util.Random;
import java.util.Scanner;

public class RockPaperScissors {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        Random random = new Random();

       
        System.out.println("Enter your choice (rock, paper, or scissors): ");
        String userChoice = scanner.nextLine();

        
        String computerChoice = random.nextInt(3) == 0 ? "rock" : random.nextInt(3) == 1 ? "paper" : "scissors";

        String winner = "";
        if (userChoice.equals(computerChoice)) {
            winner = "Tie";
        } else if (userChoice.equals("rock") && computerChoice.equals("paper")) {
            winner = "Computer";
        } else if (userChoice.equals("rock") && computerChoice.equals("scissors")) {
            winner = "User";
        } else if (userChoice.equals("paper") && computerChoice.equals("rock")) {
            winner = "User";
        } else if (userChoice.equals("paper") && computerChoice.equals("scissors")) {
            winner = "Computer";
        } else if (userChoice.equals("scissors") && computerChoice.equals("rock")) {
            winner = "Computer";
        } else if (userChoice.equals("scissors") && computerChoice.equals("paper")) {
            winner = "User";
        }

        
        System.out.println("You chose " + userChoice + ".");
        System.out.println("The computer chose " + computerChoice + ".");
        System.out.println("The winner is " + winner + ".");
    }
}
